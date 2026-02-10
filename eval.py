# -*- coding: utf-8 -*-
import os
import json
import re
import requests
import base64
import concurrent.futures
import time
import argparse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Assume eval_prompt.py is in the same directory
try:
    from eval_prompt import (
        InstructionFulfillmentPrompt, 
        AestheticQualityPrompt, 
        ConsistencyPrompt,
        ConsistencyCopyPrompt
    )
except ImportError:
    print("Warning: eval_prompt.py not found. Ensure prompt classes are available.")
    pass

# --- 1. Image Processing Functions ---
def encode_image(image_path, max_size=1024, quality=80):
    if not image_path or not os.path.exists(image_path):
        if image_path and not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
             if os.path.exists(image_path + ".png"):
                 image_path = image_path + ".png"
             else:
                 return None
        else:
            return None

    try:
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# --- 2. Text Parsing and Path Handling ---
def dataset_content_process(content: str, context_image_dir: str, eval_image_path: str):
    pattern = re.compile(r"<(image|reference_image|target_image|image_to_eval):\s*(.*?)\s*>")
    matches = pattern.findall(content)
    
    image_paths = {}
    for tag_type, filename in matches:
        full_tag = f"<{tag_type}:{filename}>"
        
        if tag_type in ["image_to_eval", "target_image"]:
            image_paths[full_tag] = eval_image_path
        else:
            name_part = filename.strip()
            if not name_part.lower().endswith(('.png', '.jpg', '.jpeg')):
                actual_filename = f"{name_part}.png"
            else:
                actual_filename = name_part
            
            image_paths[full_tag] = os.path.join(context_image_dir, actual_filename)

    all_placeholders = list(image_paths.keys())
    if not all_placeholders:
        return image_paths, [], [content]
        
    content_parts = re.split(f"({'|'.join(map(re.escape, all_placeholders))})", content)
    return image_paths, all_placeholders, content_parts

class GeminiGeneratorV1:
    def __init__(self, url, api_key, model_name="gemini-3-pro-preview"):
        self.url = f"{url}/v1beta/models/{model_name}:generateContent/"
        self.api_key = api_key

    def generate_text_core(self, user_content: str, context_image_dir: str, eval_image_path: str) -> str:
        image_paths, placeholders, content_parts = dataset_content_process(user_content, context_image_dir, eval_image_path)
        
        payload_parts = []
        for part in content_parts:
            if part in placeholders:
                encoded = encode_image(image_paths[part])
                if encoded:
                    payload_parts.append({"inlineData": {"data": encoded, "mimeType": "image/jpeg"}})
            elif part.strip():
                payload_parts.append({"text": part})
        
        if not payload_parts:
            return ""

        payload = json.dumps({
            "contents": [{"role": "user", "parts": payload_parts}],
            "generationConfig": {"responseModalities": ["TEXT"]}
        })
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        for attempt in range(5):
            try:
                response = requests.post(self.url, headers=headers, data=payload, timeout=500)
                response.raise_for_status()
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                if attempt == 2: 
                    # print(f"API Error: {e}") 
                    raise e
                time.sleep(2 * (attempt + 1)) 
        return ""


# --- 3. Visualization Utility Class ---
class Visualizer:
    @staticmethod
    def _draw_wrapped_text(draw, text, position, font, max_width, fill):
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        lines.append(' '.join(current_line))
        
        x, y = position
        for line in lines:
            draw.text((x, y), line, font=font, fill=fill)
            bbox = draw.textbbox((0, 0), line, font=font)
            y += (bbox[3] - bbox[1]) + 5
        return y

    @staticmethod
    def create_summary_images(results, output_dir, model_name, img_size=300, items_per_page=50):
        if not results: return
        os.makedirs(output_dir, exist_ok=True)
        
        text_zone_width = 800
        canvas_width = (img_size * 2) + text_zone_width + 100
        row_height = img_size + 60
        
        for page_idx in range(0, len(results), items_per_page):
            page_results = results[page_idx : page_idx + items_per_page]
            num_rows = len(page_results)
            
            canvas = Image.new('RGB', (canvas_width, num_rows * row_height), (245, 245, 245))
            draw = ImageDraw.Draw(canvas)
            
            try:
                font_path = "DejaVuSans.ttf" 
                if not os.path.exists(font_path):
                     font_path = "arial.ttf"
                
                font = ImageFont.truetype(font_path, 16)
                bold_font = ImageFont.truetype(font_path, 18)
            except:
                font = ImageFont.load_default()
                bold_font = font

            for i, item in enumerate(page_results):
                y = i * row_height
                if i % 2 == 0:
                    draw.rectangle([0, y, canvas_width, y + row_height], fill=(255, 255, 255))

                if item.get("ref_path") and os.path.exists(item["ref_path"]):
                    try:
                        with Image.open(item["ref_path"]) as img:
                            img.thumbnail((img_size, img_size))
                            canvas.paste(img, (20, y + 30))
                    except: pass
                
                if item.get("eval_path") and os.path.exists(item["eval_path"]):
                    try:
                        with Image.open(item["eval_path"]) as img:
                            img.thumbnail((img_size, img_size))
                            canvas.paste(img, (img_size + 50, y + 30))
                    except: pass
                
                text_x = (img_size * 2) + 90
                curr_y = y + 20
                
                draw.text((text_x, curr_y), f"ID: {item['id']} | Model: {model_name}", fill=(0, 0, 0), font=bold_font)
                curr_y += 30
                
                draw.text((text_x, curr_y), "Instruction:", fill=(100, 100, 100), font=bold_font)
                curr_y += 25
                instruction_text = item.get("instruction", "N/A")
                curr_y = Visualizer._draw_wrapped_text(draw, instruction_text, (text_x, curr_y), font, text_zone_width - 20, (60, 60, 60))
                curr_y += 10
                
                new_score_str = f"Evaluation: {json.dumps(item['scores'])}"
                draw.text((text_x, curr_y), new_score_str, fill=(204, 0, 0), font=bold_font)
                
                draw.line([(0, y + row_height - 1), (canvas_width, y + row_height - 1)], fill=(200, 200, 200), width=1)

            save_path = os.path.join(output_dir, f"eval_vis_{model_name}_page_{page_idx//items_per_page + 1}.jpg")
            canvas.save(save_path, "JPEG", quality=90)
            print(f"Visualization saved to: {save_path}")

# --- 4. Score Extraction Helper Function ---
def parse_single_score(text, metric_type):
    
    patterns = {
        "RC": r"Rule\s*Compliance\s*[:：]\s*(\d+)",
        "VC": r"Visual\s*Consistency\s*[:：]\s*(\d+)",
        "AQ": r"Aesthetic\s*Quality\s*[:：]\s*(\d+)",
        "Copy": r"Copy\s*[:：]\s*(\d+)"
    }
    
    p = patterns.get(metric_type)
    if not p:
        return 0
        
    # Use re.MULTILINE to ensure Copy can be handled if it is on a new line
    match = re.search(p, text, re.IGNORECASE | re.MULTILINE)
    if match:
        print(text)
        try:
            val = int(match.group(1))
            return val
        except (ValueError, IndexError):
            return 0
    return 0

# --- Core Processing Logic ---
def process_item(item, generator, builders, paths, target_metrics=None, existing_scores=None):
    """
    target_metrics: List of strings (e.g. ["Rule Compliance", "Visual Consistency"]). 
                    If None or empty, run all.
    existing_scores: Dict of previous scores (e.g. {"Rule Compliance": 8, ...})
    """
    try:
        item_id = item.get("id", "")
        context_img_dir = paths['CONTEXT_IMG_DIR']
        eval_img_dir = paths['EVAL_IMG_DIR']
        
        eval_image_path = os.path.join(eval_img_dir, f"{item_id}.png")
        rc_hint = item.get("rc_hint", "").strip()
        
        vc_hint_input = item.get("vc_hint") or item.get("VC_hint")
        
        ref_full_paths = [] 
        has_vc_task = False
        vc_hint_list = [] 

        if vc_hint_input:
            if isinstance(vc_hint_input, list):
                vc_hint_list = [h.strip() for h in vc_hint_input if h.strip()]
            else:
                vc_hint_list = [vc_hint_input.strip()]

            for single_hint in vc_hint_list:
                ref_match = re.search(r"<(image:.*?)>", single_hint)
                if ref_match:
                    ref_tag = ref_match.group(1) 
                    ref_filename = ref_tag.split(":")[-1].strip()
                    if not ref_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        ref_filename += ".png"
                    ref_full_path = os.path.join(context_img_dir, ref_filename)
                    ref_full_paths.append(ref_full_path)
                    has_vc_task = True
                else:
                    ref_full_paths.append(None)

        # Determine which metrics need to be run
        # If target_metrics is empty, run all
        should_run_rc = True
        should_run_vc = True
        should_run_aq = True

        if target_metrics and len(target_metrics) > 0:
            should_run_rc = "Rule Compliance" in target_metrics
            should_run_vc = "Visual Consistency" in target_metrics
            should_run_aq = "AestheticQuality" in target_metrics

        # Prepare old scores, default to empty dict if none
        if existing_scores is None:
            existing_scores = {}

        if not os.path.exists(eval_image_path):
            # Image does not exist, set all to 0 or reuse old values (if image is missing, old values might be problematic too)
            final_scores = {
                "Rule Compliance": existing_scores.get("Rule Compliance", 0),
                "Aesthetic Quality": existing_scores.get("Aesthetic Quality", 0)
            }
            if has_vc_task:
                final_scores["Visual Consistency"] = existing_scores.get("Visual Consistency", 0)
            
            # If running full evaluation and image doesn't exist, force to zero
            if not target_metrics:
                final_scores = {"Rule Compliance": 0, "Aesthetic Quality": 0}
                if has_vc_task: final_scores["Visual Consistency"] = 0

            info = {
                "id": item_id, 
                "scores": final_scores,
                "ref_path": ref_full_paths[0] if ref_full_paths else None,
                "eval_path": eval_image_path,
                "context": item.get("context", ""),
                "instruction": item.get("instruction", "")
            }
            
            result_item = item.copy()
            if "evaluation_text" in result_item:
                del result_item["evaluation_text"]
            result_item["scores"] = final_scores 
            return result_item, f"Warning: Eval image not found for ID {item_id}, marking as 0 (or keeping old).", info

        # Image exists, enter normal evaluation process
        raw_scores = {"RC": [], "VC": [], "AQ": []}
        
        # Only loop if there are metrics that actually need re-calculation
        need_calculation = should_run_rc or (should_run_vc and has_vc_task) or should_run_aq
        
        # --- New Logic: Plagiarism Detection (vc Copy Check) ---
        # Execute only if vc needs to run and there is a vc task
        is_plagiarism = False
        if should_run_vc and has_vc_task:
            for idx, single_hint in enumerate(vc_hint_list):
                ref_path = ref_full_paths[idx]
                if ref_path:
                    try:
                        # Build plagiarism judgment prompt
                        copy_prompt = builders["vc_copy_builder"].build(
                            reference_image=ref_path,
                            output_image=eval_image_path
                        )
                        # Call model
                        raw_copy_text = generator.generate_text_core(copy_prompt, context_img_dir, eval_image_path)
                        # Parse result using parse_single_score for consistency
                        # Expect 0 or 1; if 0, categorized as plagiarism
                        copy_score = parse_single_score(raw_copy_text, "Copy")
                        
                        if copy_score == 0:
                            is_plagiarism = True
                            # If even one is judged as plagiarism, break; subsequent vc results will be 0
                            break
                    except Exception:
                        # If error occurs during detection, continue the flow
                        pass

        if need_calculation:
            for _ in range(3):
                # --- IF Dimension ---
                if should_run_rc:
                    rc_prompt = builders["rc_builder"].build(hint=rc_hint, output_image=eval_image_path)
                    raw_rc_text = generator.generate_text_core(rc_prompt, context_img_dir, eval_image_path)
                    raw_scores["RC"].append(parse_single_score(raw_rc_text, "RC"))
                
                # --- VC Dimension ---
                if should_run_vc and has_vc_task:
                    # If plagiarism is detected, give 0 directly, skip prompt calls
                    if is_plagiarism:
                        raw_scores["VC"].append(0)
                    else:
                        sub_vc_scores = []
                        for idx, single_hint in enumerate(vc_hint_list):
                            ref_path = ref_full_paths[idx]
                            if ref_path:
                                vc_prompt = builders["vc_builder"].build(
                                    reference_image=ref_path, 
                                    hint=single_hint, 
                                    output_image=eval_image_path
                                )
                                raw_vc_text = generator.generate_text_core(vc_prompt, context_img_dir, eval_image_path)
                                sub_vc_scores.append(parse_single_score(raw_vc_text, "VC"))
                            else:
                                sub_vc_scores.append(0)
                        iter_avg_vc = sum(sub_vc_scores) / len(sub_vc_scores)
                        raw_scores["VC"].append(iter_avg_vc)
                elif should_run_vc:
                    raw_scores["VC"].append(0)

                # --- AQ Dimension ---
                if should_run_aq:
                    aq_prompt = builders["aq_builder"].build(output_image=eval_image_path)
                    raw_aq_text = generator.generate_text_core(aq_prompt, context_img_dir, eval_image_path)
                    raw_scores["AQ"].append(parse_single_score(raw_aq_text, "AQ"))
        
        def calc_avg(score_list):
            if not score_list: return 0
            return int(round(sum(score_list) / len(score_list)))

        # Combine final scores: Use new score if run, otherwise use old score
        final_scores = {}

        # Rule Compliance
        if should_run_rc:
            final_scores["Rule Compliance"] = calc_avg(raw_scores["RC"])
        else:
            final_scores["Rule Compliance"] = existing_scores.get("Rule Compliance", 0)

        # Aesthetic Quality
        if should_run_aq:
            final_scores["Aesthetic Quality"] = calc_avg(raw_scores["AQ"])
        else:
            final_scores["Aesthetic Quality"] = existing_scores.get("Aesthetic Quality", 0)

        # Consistency
        if has_vc_task:
            if should_run_vc:
                final_scores["Visual Consistency"] = calc_avg(raw_scores["VC"])
            else:
                final_scores["Visual Consistency"] = existing_scores.get("Visual Consistency", 0)

        info = {
            "id": item_id, 
            "scores": final_scores,
            "ref_path": ref_full_paths[0] if ref_full_paths else None, 
            "eval_path": eval_image_path,
            "context": item.get("context", ""),
            "instruction": item.get("instruction", "")
        }
        
        result_item = item.copy()
        if "evaluation_text" in result_item:
            del result_item["evaluation_text"]
        result_item["scores"] = final_scores 
        
        return result_item, None, info

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error on ID {item.get('id')}: {str(e)}", None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMICL Eval Script")
    parser.add_argument("--dimension", required=True, help="Dimension name (e.g., dimension_conflict)")
    parser.add_argument("--model", required=True, help="Target Model name being evaluated (e.g., nanobanana2)")
    parser.add_argument("--base_path", required=True, help="Base path for MMICL-dev")
    
    # Optional arguments
    parser.add_argument("--max_workers", type=int, default=20, help="Number of concurrent threads")
    parser.add_argument("--url", default="", help="Gemini API Base URL")
    parser.add_argument("--api_key", default="", help="Gemini API Key")
    parser.add_argument("--api_model_name", default="gemini-3-pro-preview", help="Gemini Model Version (e.g., gemini-3-pro-preview)")
    
    # Metric argument
    parser.add_argument("--metric", default="", help="Specific metrics to update, separated by comma if needed (e.g. 'Visual Consistency' or 'Rule Compliance,AestheticQuality')")

    args = parser.parse_args()
    
    BASE_PATH = args.base_path
    DIMENSION = args.dimension
    MODEL_NAME = args.model 
    
    MAX_WORKERS = args.max_workers
    API_URL = args.url
    API_KEY = args.api_key
    API_MODEL_NAME = args.api_model_name
    METRIC_ARG = args.metric.strip()

    # --- Path Configuration ---
    DATASET_DIR = os.path.join(BASE_PATH, "dataset", DIMENSION)
    INPUT_JSON = os.path.join(DATASET_DIR, "test_data.json")
    CONTEXT_IMG_DIR = os.path.join(DATASET_DIR, "images")
    
    EVAL_IMG_DIR = os.path.join(BASE_PATH, "outputs", MODEL_NAME, DIMENSION)
    
    OUTPUT_EVAL_DIR = os.path.join(BASE_PATH, "evaluation", DIMENSION)
    os.makedirs(OUTPUT_EVAL_DIR, exist_ok=True)
    OUTPUT_JSON = os.path.join(OUTPUT_EVAL_DIR, f"eval_results_{MODEL_NAME}.json")
    
    OUTPUT_VIS_DIR = os.path.join(BASE_PATH, "visual", DIMENSION)
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

    paths_config = {
        'CONTEXT_IMG_DIR': CONTEXT_IMG_DIR,
        'EVAL_IMG_DIR': EVAL_IMG_DIR
    }

    # Parse Metrics to update
    target_metrics = []
    if METRIC_ARG:
        # Support partial matching and multiple selections
        possible_metrics = ["Rule Compliance", "AestheticQuality", "Visual Consistency"]
        for pm in possible_metrics:
            if pm in METRIC_ARG:
                target_metrics.append(pm)
        if not target_metrics:
            print(f"Warning: Metric argument '{METRIC_ARG}' provided but no known metrics matched. Will run default (ALL).")

    print(f"--- Configuration ---")
    print(f"Dimension: {DIMENSION}")
    print(f"Target Model: {MODEL_NAME}")
    print(f"Evaluator API Model: {API_MODEL_NAME}")
    print(f"Concurrency: {MAX_WORKERS}")
    print(f"Input JSON: {INPUT_JSON}")
    print(f"Output JSON: {OUTPUT_JSON}")
    if target_metrics:
        print(f"Target Metrics to Update: {target_metrics}")
    else:
        print(f"Target Metrics: ALL (New Run)")
    print(f"---------------------")

    if not os.path.exists(INPUT_JSON):
        print(f"Error: Dataset JSON not found at {INPUT_JSON}")
        exit(1)
    if not os.path.exists(EVAL_IMG_DIR):
        print(f"Warning: Eval image directory not found at {EVAL_IMG_DIR}. Proceeding, but images may be missing.")

    # Prepare existing score data (if in update mode)
    existing_data_map = {}
    if target_metrics and os.path.exists(OUTPUT_JSON):
        print(f"Loading existing results from {OUTPUT_JSON} for update...")
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                old_results = json.load(f)
                for item in old_results:
                    if "id" in item:
                        existing_data_map[item["id"]] = item.get("scores", {})
        except Exception as e:
            print(f"Error reading existing JSON: {e}. Will proceed as if empty.")

    # Initialize Generator
    gemini_generator = GeminiGeneratorV1(
        url=API_URL,
        api_key=API_KEY,
        model_name=API_MODEL_NAME
    )

    builders = {
        "rc_builder": InstructionFulfillmentPrompt(),
        "aq_builder": AestheticQualityPrompt(),
        "vc_builder": ConsistencyPrompt(),
        "vc_copy_builder": ConsistencyCopyPrompt()
    }

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    final_results = []
    vis_data_list = []
    total_processed = 0

    print(f"Starting evaluation of {len(data_list)} samples...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {}
        for item in data_list:
            item_id = item.get("id")
            existing_scores = existing_data_map.get(item_id, None)
            
            future = executor.submit(
                process_item, 
                item, 
                gemini_generator, 
                builders, 
                paths_config, 
                target_metrics, 
                existing_scores
            )
            future_to_item[future] = item
        
        for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(data_list), desc="Evaluating"):
            result, log, info = future.result()
            if result:
                final_results.append(result)
                vis_data_list.append(info) 
                total_processed += 1
            if log:
                if "Error" in log:
                    print(f"\n{log}")


    final_results.sort(key=lambda x: int(x['id']) if isinstance(x['id'], int) or x['id'].isdigit() else x['id'])

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print("\nGenerating summary visualization images...")
    Visualizer.create_summary_images(vis_data_list, OUTPUT_VIS_DIR, MODEL_NAME)

    print(f"\nResults saved to: {OUTPUT_JSON}")
    print(f"Visualization images saved in: {OUTPUT_VIS_DIR}")
    