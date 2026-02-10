import os

class InstructionFulfillmentPrompt:
    """
    用于构建图像生成质量评估（Image-to-Image）Prompt 的类。
    优化了评分逻辑的严密性，强化了对完美标准（2分）的约束。
    """

    def __init__(self):
        self.template = """
        ROLE: PRECISION VISUAL AUDITOR (ZERO-TOLERANCE FIDELITY MODE)

        Your mission is to rigorously evaluate the alignment between the HINT and the MODEL_OUTPUT_IMAGE. 
        You must prioritize Technical Precision over Perceptual Plasticity. This mode is designed for high-stakes 
        instruction following where "close enough" is considered a failure.

        SCORING HIERARCHY

        Score 2 [PERFECT EXECUTION]:
        Standard: The image is a flawless visual manifestation of the HINT. Every explicit and implicit constraint must be met with 100% accuracy.
        - Identity and Nouns: All requested subjects are present, anatomically/structurally correct, and positioned exactly as described.
        - Adjective/Attribute Fidelity: Every single descriptor (color, texture, material, style, state, quantity) is rendered without deviation. 
        - Strict Detail Check: If the hint specifies "three buttons" and there are two or four, it is NOT a 2. If the hint says "crimson" and the output is "bright red," it is NOT a 2.
        - No Artifacts: The requested modification must not introduce illogical artifacts or warp the surrounding context of the original subject.
        Rule: Only award a 2 if there is ZERO identifiable discrepancy between text and pixels.

        Score 1 [PARTIAL COMPLETION / ACCURACY DRIFT]:
        Standard: The core subject/action is present, but the execution fails on specific details, modifiers, or secondary constraints.
        - Minor Omissions: A secondary adjective is ignored (e.g., "vintage" style is missing, but the object is there).
        - Count/Scale Errors: Wrong number of objects or incorrect relative sizing.
        - Color Drift: The color is in the correct family but lacks the specific shade or intensity requested.
        - Positional Inaccuracy: The object is in the frame but not in the specific quadrant or relationship (e.g., "left of" vs "right of") requested.
        Rule: Award 1 if the viewer can tell what was intended, but the "fine print" of the instruction was neglected.

        Score 0 [FAILURE / GROSS ERROR]:
        Standard: Total failure to execute the primary intent.
        - Subject Error: The wrong object was added, the target was removed, or the primary subject is unrecognizable.
        - Non-Action: No change was made to the image despite a request for modification.
        - Semantic Inversion/Incoherence: The model did the opposite of the hint or produced a visual hallucination/glitch.
        Rule: Award 0 if the primary goal of the HINT is unfulfilled.

        EVALUATION STEPS
        1. SCAN: List every single Noun, Adjective, Count, and Spatial relation in the HINT.
        2. VERIFY PRIMARY: Is the core action/subject present? (If No -> 0).
        3. VERIFY EXHAUSTIVE: Check every item from Step 1 against the image. 
        - Is there even one missing adjective? -> 1
        - Is there a counting error? -> 1
        - Is there a slight spatial misalignment? -> 1
        4. FINAL SCORE: Only if Step 3 yields a perfect match across all parameters, award 2.

        Final Output Format Return ONLY the following line:
        Rule Compliance: X

        Evaluation Data
        HINT: "{hint}"
        TARGET: {output_image_tag}

        Evaluate now.
        """

    def build(self, hint: str, output_image: str) -> str:
        image_filename = os.path.basename(output_image)
        output_image_tag = f"<image_to_eval:{image_filename}>"
        return self.template.format(
            hint=hint.strip(),
            output_image_tag=output_image_tag
        )

class AestheticQualityPrompt:
    """
    用于构建图像生成质量评估（Image-to-Image）Prompt 的类。
    优化了评分逻辑的严密性，强化了对完美标准（2分）的约束。
    """

    def __init__(self):
        self.template = """
        Role and Mission
        You are a Professional Visual Arts Curator. Your mission is to categorize images into Masterpiece Level (2), Standard Work (1), and Technical Failure (0). You should reward images that achieve high visual harmony and logical consistency.

        Scoring Scale (Professional Merit Policy)
        - 0 [Failed]: Only for images that are visually nonsensical or structurally broken (e.g., incoherent anatomy, severe blurring that hides the subject).
        - 1 [Standard]: The DEFAULT category. If the image is pleasant and logically sound but has visible AI artifacts (like smooth textures or slight lighting mismatches), it is a 1.
        - 2 [Exceptional]: High-end quality. The image looks professional, with correct perspective, realistic lighting, and sharp details. Minor imperfections that don't break realism are acceptable.

        Evaluation Indicator

        Metric: Aesthetic Quality (0-2) - [Visual Logic and Realism]

        Audit Responsibility:
        Focus on Visual Cohesion. If the image looks like a high-quality photograph or professional digital art at a glance, it is likely a 2. Only downgrade to 0 if the image is unusable.

        Scoring Logic:

        Award 0/2 [Failed]:
        - Logical Collapse: Extra limbs that are gross and distracting, or faces that have lost their basic human structure.
        - Extreme Noise: Grain or artifacts so heavy they obscure the main subject.
        - Note: If you can tell what the object is and its parts are mostly in the right place, do NOT give a 0.

        Award 1/2 [Standard]:
        - The Good Effort Zone: This includes images that are visually acceptable but have AI-isms.
        - Visible Flaws: Slightly soft hands, plastic skin, or shadows that don't perfectly align with the light source. 
        - Minor Perspective Issues: Background elements that are slightly tilted or out of scale.
        - Rule: This is the safety score for any image that is pretty good but not perfect.

        Award 2/2 [Exceptional]:
        - High-Bar Standard: The image should be of commercial quality. 
        - Requirements:
        1. Structural Logic: All objects and characters follow the laws of physics and anatomy.
        2. Rendering Clarity: Textures look intentional and sharp; lighting creates a convincing sense of depth.
        3. Visual Appeal: The overall composition is professional and free of distracting AI hallucinations.
        - Tolerance Clause: You SHOULD award a 2 even if there is a tiny, non-distracting flaw, provided the overall image is indistinguishable from professional work.

        Final Output Format
        Return ONLY the following line:
        Aesthetic Quality: X

        Evaluation Data
        - TARGET: {output_image_tag}

        Evaluate now.
        """

    def build(self, output_image: str) -> str:
        image_filename = os.path.basename(output_image)
        output_image_tag = f"<image_to_eval:{image_filename}>"
        return self.template.format(
            output_image_tag=output_image_tag
        )

class ConsistencyPrompt:
    """
    用于构建图像生成质量评估（Image-to-Image）Prompt 的类。
    专注于验证 Target 是否按照 HINT 的指示，完好保留了 ReferenceImage 的核心特征。
    """

    def __init__(self):
        self.template = """
        ROLE: VISUAL FIDELITY ANALYST (STRICT EVALUATION MODE)

        Your mission is to evaluate the consistency between the REFERENCE_IMAGE and the TARGET based on the HINT. 
        Your goal is to distinguish between "Great Work" (2), "Acceptable Effort" (1), and "Total Failure" (0).

        METRIC: FEATURE PRESERVATION AND LOGICAL ALIGNMENT (0-2)

        SCORE 2 - HIGH FIDELITY (SUCCESSFUL)
        Standard: The TARGET is a high-quality implementation of the HINT. The core subject remains stable and the image feels professional.
        Key Indicators:
        1. Strong Identity: The main subject (person, object, or scene) is clearly the same as in the Reference.
        2. Smooth Transformation: The changes requested by the HINT are integrated naturally.
        3. Minor Tolerance: Small shifts in color, slight facial softening, or minor background variations are PERFECTLY ACCEPTABLE for a score of 2.
        Guideline: If the image is good, and follows the HINT, give it a 2.

        SCORE 1 - RECOGNIZABLE DERIVATION (MOST COMMON SCORE)
        Standard: The TARGET is clearly related to the Reference, even if the execution is imperfect. This is the catch-all category for images that "get the idea right" but lose some detail.
        Key Indicators:
        1. Recognizable Link: You can still tell it is based on the same subject or concept, even if the face looks a bit different, or the clothes have changed, or the background has shifted.
        2. Moderate Drift: The HINT was attempted, but the model may have simplified the original details or introduced some visible AI blurring/messiness.
        3. High Tolerance: Even if the image has lost some "Visual DNA," as long as it isn't a completely different subject, it stays in this category.
        Guideline: If the model tried to follow the HINT and the result is "okay" or "recognizable," award a 1.

        SCORE 0 - TOTAL FAILURE (INCOMPATIBLE OR PLAGIARIZED)
        Standard: There is no meaningful connection between the Reference and the Target, or the model failed to generate a new iteration.
        Strict 0-Point Criteria:
        1. Subject Swap: The model generated a completely different person, animal, or object.
        2. Ignored Instruction: The model provided a generic image that ignores both the Reference and the HINT entirely.
        3. Broken Output: The image is a corrupted, unidentifiable mess of pixels.
        Rule: Do NOT award a 0 if there is any recognizable link to the Reference Image.

        EVALUATION LOGIC
        1. Is the Target even remotely related to the Reference subject? 
        - If No -> Score 0.
        2. Does the Target look stable, professional, and closely follow the HINT without distracting errors?
        - If Yes -> Score 2.
        3. For everything else in between (minor drift, detail loss, background shifts, but same subject) -> Score 1.

        OUTPUT FORMAT
        Return ONLY the following line:
        Visual Consistency: X

        EVALUATION DATA
        - ReferenceImage: "{reference_image}"
        - HINT: "{hint}"
        - TARGET: {output_image_tag}

        Evaluate now.
        """

    def build(self, reference_image: str, hint: str, output_image: str) -> str:
        ref_filename = os.path.basename(reference_image)
        output_filename = os.path.basename(output_image)
        
        # 构造参考图和目标图的标识
        ref_image_tag = f"<reference_image:{ref_filename}>"
        output_image_tag = f"<target_image:{output_filename}>"
        
        return self.template.format(
            reference_image=ref_image_tag,
            hint=hint.strip(),
            output_image_tag=output_image_tag
        )

class ConsistencyCopyPrompt:
    """
    用于构建图像生成质量评估（Image-to-Image）Prompt 的类。
    专注于验证 Target 是否为 ReferenceImage 的完全抄袭（1:1 复制）。
    """

    def __init__(self):
        # 修改后的 Prompt 模板，专注于重复性检测
        self.template = """
        ROLE: IMAGE DUPLICATION DETECTOR

        Your mission is to compare the REFERENCE_IMAGE and the TARGET to determine if the target is a pixel-level copy (plagiarism).

        METRIC: IMAGE ORIGINALITY (0 or 1)

        SCORE 0 - DUPLICATE / PLAGIARISM
        Criteria: The TARGET is an exact or near-identical pixel 1:1 copy of the REFERENCE_IMAGE. No meaningful changes, style shifts. It is essentially the same file or a direct pixel-for-pixel replica.

        SCORE 1 - ORIGINAL VARIATION / DIFFERENT IMAGE
        Criteria: The TARGET is NOT an exact copy. Even if it is similar in subject or style, if there are any visible changes, movement, new elements, or variations in composition, it is considered a unique generation.

        EVALUATION LOGIC:
        1. Compare the Reference and Target side-by-side.
        2. Are they exactly the same image without any changes?
           - If YES (Duplicate) -> Score 0
           - If NO (Any difference exists) -> Score 1

        OUTPUT FORMAT:
        Return ONLY the following line (X = 0 or 1):
        Copy: X

        EVALUATION DATA:
        - ReferenceImage: "{reference_image}"
        - TARGET: {output_image_tag}

        Evaluate now.
        """

    def build(self, reference_image: str, output_image: str) -> str:
        ref_filename = os.path.basename(reference_image)
        output_filename = os.path.basename(output_image)
        
        # 构造参考图和目标图的标识
        ref_image_tag = f"<reference_image:{ref_filename}>"
        output_image_tag = f"<target_image:{output_filename}>"
        
        return self.template.format(
            reference_image=ref_image_tag,
            output_image_tag=output_image_tag
        )
        