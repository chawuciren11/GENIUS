import json
import os

def calculate_metric_stats(dataset_path, eval_path, dimension_name):
    """
    Calculate statistics for a single sub-dimension
    """
    # 1. Load files
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        with open(eval_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: File not found {e.filename}")
        return None

    # 2. Build score lookup table (Key: ID, Value: Scores)
    eval_dict = {item['id']: item.get('scores', {}) for item in eval_data}

    # Initialize statistical variables
    stats = {
        "count_total": 0,           # Total number of valid matched data
        "sum_inst": 0, "n_inst": 0, # Rule Compliance
        "sum_cons": 0, "n_cons": 0, # Visual Consistency  
        "sum_aes": 0, "n_aes": 0    # Aesthetic Quality
    }

    # 3. Traverse Dataset (based on data in dataset file)
    for item in dataset:
        item_id = item.get('id')
        
        # Calculate only if the corresponding ID is found in the evaluation file
        if item_id in eval_dict:
            scores = eval_dict[item_id]
            stats["count_total"] += 1
            
            # --- Rule Compliance ---
            if "Rule Compliance" in scores:
                raw = scores["Rule Compliance"]
                stats["sum_inst"] += raw * 50  # 0->0, 1->50, 2->100
                stats["n_inst"] += 1

            # --- Visual Consistency (handling potential typos) ---
            val_cons = None
            if "Visual Consistency" in scores:
                # val_cons = scores["Visual Consistency "] * 0.5 * scores["Rule Compliance"]
                val_cons = scores["Visual Consistency"]
            elif "Consisitency" in scores:
                val_cons = scores["Consisitency"]
            
            if val_cons is not None:
                stats["sum_cons"] += val_cons * 50
                stats["n_cons"] += 1

            # --- Aesthetic Quality ---
            if "Aesthetic Quality" in scores:
                raw = scores["Aesthetic Quality"]
                stats["sum_aes"] += raw * 50
                stats["n_aes"] += 1

    # 4. Calculate average scores for this sub-dimension (for display)
    # If the count for a metric is 0, return 0 or None
    avg_inst = stats["sum_inst"] / stats["n_inst"] if stats["n_inst"] > 0 else 0
    avg_cons = stats["sum_cons"] / stats["n_cons"] if stats["n_cons"] > 0 else None
    avg_aes  = stats["sum_aes"] / stats["n_aes"] if stats["n_aes"] > 0 else 0

    return {
        "name": dimension_name,
        "total_items": stats["count_total"],
        "avg_inst": avg_inst,
        "avg_cons": avg_cons,
        "avg_aes": avg_aes,
        "raw_stats": stats # Return raw stats for global weighted calculation
    }

def main(base_path="", model_names=["nanobanana2_edit"]):
    # Define base file path templates
    # Use {model} as a placeholder for subsequent replacement
    base_pairs = [
        (
            "{base_dir}/dataset/prior_conflicting/test_data.json",
            "{base_dir}/evaluation/prior_conflicting/eval_results_{model}.json",
            "prior_conflicting"
        ),
        (
            "{base_dir}/dataset/multi_semantic/test_data.json",
            "{base_dir}/evaluation/multi_semantic/eval_results_{model}.json",
            "multi_semantic"
        ),
        (
            "{base_dir}/dataset/implicit_pattern/test_data.json",
            "{base_dir}/evaluation/implicit_pattern/eval_results_{model}.json",
            "implicit_pattern"
        ),
        (
            "{base_dir}/dataset/symbolic_constraint/test_data.json",
            "{base_dir}/evaluation/symbolic_constraint/eval_results_{model}.json",
            "symbolic_constraint"
        ),
        (
            "{base_dir}/dataset/visual_constraint/test_data.json",
            "{base_dir}/evaluation/visual_constraint/eval_results_{model}.json",
            "visual_constraint"
        )
    ]

    for model_name in model_names:
        print("\n" + "="*80)
        print(f"Calculating Model: {model_name}")
        print("="*80)

        # Generate actual pairs paths based on the current model name
        current_pairs = [
            (d_p.format(base_dir=base_path), e_p.format(base_dir=base_path, model=model_name), name) 
            for d_p, e_p, name in base_pairs
        ]

        results = []
        
        # Global statistics accumulators (for calculating data-weighted total scores)
        global_stats = {
            "sum_inst": 0, "n_inst": 0,
            "sum_cons": 0, "n_cons": 0,
            "sum_aes": 0,  "n_aes": 0,
            "total_data_count": 0
        }

        print(f"{'Sub-Dimension':<20} | {'Count':<6} | {'RC(%)':<10} | {'VC(%)':<10} | {'AQ(%)':<10}")
        print("-" * 75)

        # 1. Process sub-dimensions one by one
        for d_path, e_path, name in current_pairs:
            res = calculate_metric_stats(d_path, e_path, name)
            if res:
                results.append(res)
                
                # Format output for individual sub-dimension results
                inst_str = f"{res['avg_inst']:.2f}"
                cons_str = f"{res['avg_cons']:.2f}" if res['avg_cons'] is not None else "N/A"
                aes_str  = f"{res['avg_aes']:.2f}"
                
                print(f"{name:<20} | {res['total_items']:<6} | {inst_str:<10} | {cons_str:<10} | {aes_str:<10}")

                # Accumulate to global statistics
                raw = res['raw_stats']
                global_stats["total_data_count"] += raw["count_total"]
                global_stats["sum_inst"] += raw["sum_inst"]
                global_stats["n_inst"] += raw["n_inst"]
                global_stats["sum_cons"] += raw["sum_cons"]
                global_stats["n_cons"] += raw["n_cons"]
                global_stats["sum_aes"] += raw["sum_aes"]
                global_stats["n_aes"] += raw["n_aes"]

        print("-" * 75)

        # 2. Calculate global weighted average scores (Weighted Average across all dimensions)
        final_global_inst = global_stats["sum_inst"] / global_stats["n_inst"] if global_stats["n_inst"] > 0 else 0
        final_global_cons = global_stats["sum_cons"] / global_stats["n_cons"] if global_stats["n_cons"] > 0 else 0
        final_global_aes  = global_stats["sum_aes"]  / global_stats["n_aes"]  if global_stats["n_aes"]  > 0 else 0

        print(f"{'All Dimensions (Weighted)':<20} | {global_stats['total_data_count']:<6} | {final_global_inst:<10.2f} | {final_global_cons:<10.2f} | {final_global_aes:<10.2f}")

        # 3. Calculate final weighted score (0.6 : 0.35 : 0.05)
        final_score = (0.6 * final_global_inst) + (0.35 * final_global_cons) + (0.05 * final_global_aes)

        print("\n" + "="*40)
        print(f"Final Results Summary [{model_name}]")
        print("="*40)
        print(f"Total Data Count: {global_stats['total_data_count']}")
        print("-" * 40)
        print(f"Metric Average Scores (Data-Weighted):")
        print(f"  Rule Compliance      : {final_global_inst:.2f}")
        print(f"  Visual Consistency   : {final_global_cons:.2f}")
        print(f"  Aesthetic Quality    : {final_global_aes:.2f}")
        print("-" * 40)
        print(f"Weighted Total Score   : {final_score:.4f}")
        print("Formula: 0.6 * RC + 0.35 * VC + 0.05 * AQ")
        print("="*40)

if __name__ == "__main__":
    # Define the list of models you need to run here
    models = ["nanobanana2"]
    base_path = ""
    main(base_path, models)
    