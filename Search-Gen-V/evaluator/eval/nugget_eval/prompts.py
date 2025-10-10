"""Prompt templates for nugget matching evaluation based on template.md"""

from typing import List, Dict, Any


class PromptTemplateManager:
    """Manager for different prompt templates with format support"""
    
    def __init__(self):
        self.format_instructions = self._initialize_format_instructions()
    
    def _initialize_format_instructions(self) -> Dict[str, str]:
        """Initialize format-specific instructions"""
        return {
            "json": (
                'Respond with a JSON array containing exactly one label for each nugget:\n'
                '["support", "partial_support", "not_support"]\n'
                'Example: ["support", "not_support", "partial_support"]'
            ),
            "csv": (
                'Respond with comma-separated values, one label for each nugget:\n'
                'support,partial_support,not_support\n'
                'Example: support,not_support,partial_support'
            ),
            "python_list": (
                'Respond with a Python list containing exactly one label for each nugget:\n'
                "['support', 'partial_support', 'not_support']\n"
                "Example: ['support', 'not_support', 'partial_support']"
            ),
            "yaml": (
                'Respond with a YAML list, one label for each nugget:\n'
                '- support\n'
                '- partial_support\n'
                '- not_support\n'
                'Example:\n- support\n- not_support\n- partial_support'
            ),
            "markdown": (
                'Respond with a Markdown unordered list, one label for each nugget:\n'
                '* support\n'
                '* partial_support\n'
                '* not_support\n'
                'Example:\n* support\n* not_support\n* partial_support'
            ),
            "xml": (
                'Respond with XML format, one label for each nugget:\n'
                '<labels>\n'
                '  <label>support</label>\n'
                '  <label>partial_support</label>\n'
                '  <label>not_support</label>\n'
                '</labels>\n'
                'Example:\n<labels>\n  <label>support</label>\n  <label>not_support</label>\n</labels>'
            ),
            "tsv": (
                'Respond with tab-separated values, one label for each nugget:\n'
                'support\tpartial_support\tnot_support\n'
                'Example: support\tnot_support\tpartial_support'
            ),
            "numbered": (
                'Respond with a numbered list, one label for each nugget:\n'
                '1. support\n'
                '2. partial_support\n'
                '3. not_support\n'
                'Example:\n1. support\n2. not_support\n3. partial_support'
            ),
            "comma_separated": (
                'Respond with comma-separated values with spaces, one label for each nugget:\n'
                'support, partial_support, not_support\n'
                'Example: support, not_support, partial_support'
            ),
            "pipe_separated": (
                'Respond with pipe-separated values, one label for each nugget:\n'
                'support|partial_support|not_support\n'
                'Example: support|not_support|partial_support'
            ),
            "adaptive": (
                'Respond using ANY of these formats (choose the one that works best for you):\n'
                '1. JSON array: ["support", "partial_support", "not_support"]\n'
                '2. Markdown list: * support\n* partial_support\n* not_support\n'
                '3. Comma-separated: support, partial_support, not_support\n'
                '4. Numbered list: 1. support\n2. partial_support\n3. not_support\n'
                '\nProvide exactly one label for each nugget in the same order.'
            )
        }
    
    def _get_format_instruction(self, format_type: str) -> str:
        """Get format-specific output instruction"""
        return self.format_instructions.get(format_type, self.format_instructions["adaptive"])
    
    def _get_no_reasoning_system(self, format_type: str) -> str:
        """System prompt for no reasoning mode"""
        return """You are NuggetMatchJudge.

Task:
Given a search query, a passage, and {num_nuggets} nuggets, assign one label to each nugget: "support", "partial_support", or "not_support".

Evidence rules (use the passage only):
- support: The passage explicitly and sufficiently affirms the nugget's key facts relevant to the query; no contradiction.
- partial_support: The passage contains directly relevant but incomplete/hedged/ambiguous information; minor inference is needed; at least one essential part is present.
- not_support: The nugget is absent, contradicted, or requires external knowledge beyond the passage.

Labeling checklist (apply per nugget):
- support (all must hold):
  - All essential facts in the nugget are explicitly present or clearly paraphrased in the passage.
  - No contradiction anywhere in the passage.
  - Qualifiers match: time, modality, scope, quantities, and conditions align with the nugget.
  - No external knowledge is required beyond the passage.
- partial_support (all must hold):
  - At least one essential fact is directly supported, but at least one other essential element is missing, hedged, ambiguous, or unspecified.
  - No direct contradiction with the nugget's claim.
  - Only minor inference is needed (e.g., synonyms or near-paraphrases); not speculative.
- not_support (any one suffices):
  - The nugget's key facts are absent or only topically related without direct support.
  - The passage contradicts the nugget.
  - Nontrivial external or world knowledge is required to establish the nugget.
  - Key qualifiers mismatch (time, modality, scope, quantities, or conditions).

Decision policy:
- Base the decision on verifiable snippets from the passage (identify them internally; do NOT output them).
- If information is missing or contradicted, choose "not_support".
- If some relevant evidence exists but is incomplete, choose "partial_support".
- Paraphrases and synonymous formulations count as support if the meaning clearly aligns.

Output constraints:
- {format_instruction}
- Length must equal {num_nuggets}, order matches the input nuggets.
- Elements must be one of: "support", "partial_support", "not_support"."""

    def _get_no_reasoning_user(self, format_type: str) -> str:
        """User prompt for no reasoning mode with format support"""
        format_instruction = self._get_format_instruction(format_type)
        return """Search Query: {query}
Passage: {passage}
Nuggets: {nugget_list}

{format_instruction}

Labels:"""

    def _get_long_cot_system(self, format_type: str) -> str:
        """System prompt for long chain-of-thought mode"""
        return """You are NuggetMatchJudge.

Task:
Given a search query, a passage, and {num_nuggets} nuggets, assign one label to each nugget: "support", "partial_support", or "not_support".

Definitions (evidence from the passage only):
- support: The passage explicitly and sufficiently affirms the nugget's key facts relevant to the query; no contradiction.
- partial_support: The passage contains directly relevant but incomplete/hedged/ambiguous information; minor inference is needed; at least one essential part is present.
- not_support: The nugget is absent, contradicted, or requires external knowledge beyond the passage.

 Labeling checklist (apply per nugget):
 - support (all must hold):
   - All essential facts in the nugget are explicitly present or clearly paraphrased in the passage.
   - No contradiction anywhere in the passage.
   - Qualifiers match: time, modality, scope, quantities, and conditions align with the nugget.
   - No external knowledge is required beyond the passage.
 - partial_support (all must hold):
   - At least one essential fact is directly supported, but at least one other essential element is missing, hedged, ambiguous, or unspecified.
   - No direct contradiction with the nugget's claim.
   - Only minor inference is needed (e.g., synonyms or near-paraphrases); not speculative.
 - not_support (any one suffices):
   - The nugget's key facts are absent or only topically related without direct support.
   - The passage contradicts the nugget.
   - Nontrivial external or world knowledge is required to establish the nugget.
   - Key qualifiers mismatch (time, modality, scope, quantities, or conditions).

Rules:
- Use only the passage; ignore external knowledge.
- Base judgments on verifiable snippets from the passage; quote the minimal necessary phrase(s).
- If information is missing or contradicted → not_support.
- If some relevant evidence exists but is incomplete → partial_support.
- Paraphrases count as support if the meaning clearly aligns.

Output format (three-part):
1) Optional long-form chain-of-thought: If you are a long-CoT model (e.g., DeepSeek-R1, Qwen3), place any free-form internal reasoning ONLY inside a single <think> ... </think> block. This block may be arbitrarily long. Do NOT include square brackets [] or any JSON-like structures inside <think>. If you do not need it, omit the block entirely.
2) Structured reasoning: Wrap ALL nugget-level structured reasoning in a single block delimited by <reasoning> and </reasoning>. Inside the block, for each nugget k = 1..{num_nuggets}, output a standardized multi-line item with the four fields in order: nugget, snippet, rationale, decision. Use the following format for every k:
   <reasoning>
   k.
     nugget="<nugget_k>"
     snippet="<verbatim phrase(s) from passage>"
     rationale="<free-form reasoning; one or more sentences/clauses>"
     decision="<support|partial_support|not_support>"
   ...
   </reasoning>
   - The rationale may be any length/format (short clause, sentence(s), or bullet-like clauses separated by ';' or new lines). Do NOT include square brackets [] inside <reasoning>. Do NOT output any other tags outside <think> ... </think> and <reasoning> ... </reasoning>.
3) Final answer: The LAST part of your response must be the labels in the requested format.
   - {format_instruction}
   - No extra text after the labels.
   - Count must equal {num_nuggets}."""

    def _get_long_cot_user(self, format_type: str) -> str:
        """User prompt for long chain-of-thought mode"""
        return """Search Query: {query}
Passage: {passage}
Nuggets ({num_nuggets}): {nugget_list}

Begin reasoning and final labels:"""

    def _get_short_cot_system(self, format_type: str) -> str:
        """System prompt for short chain-of-thought mode"""
        return """You are NuggetMatchJudge.

Task:
Given a search query, a passage, and {num_nuggets} nuggets, assign one label to each nugget: "support", "partial_support", or "not_support".

Definitions (evidence from the passage only):
- support: The passage explicitly and sufficiently affirms the nugget's key facts relevant to the query; no contradiction.
- partial_support: The passage contains directly relevant but incomplete/hedged/ambiguous information; minor inference is needed; at least one essential part is present.
- not_support: The nugget is absent, contradicted, or requires external knowledge beyond the passage.

 Labeling checklist (apply per nugget):
 - support (all must hold):
   - All essential facts in the nugget are explicitly present or clearly paraphrased in the passage.
   - No contradiction anywhere in the passage.
   - Qualifiers match: time, modality, scope, quantities, and conditions align with the nugget.
   - No external knowledge is required beyond the passage.
 - partial_support (all must hold):
   - At least one essential fact is directly supported, but at least one other essential element is missing, hedged, ambiguous, or unspecified.
   - No direct contradiction with the nugget's claim.
   - Only minor inference is needed (e.g., synonyms or near-paraphrases); not speculative.
 - not_support (any one suffices):
   - The nugget's key facts are absent or only topically related without direct support.
   - The passage contradicts the nugget.
   - Nontrivial external or world knowledge is required to establish the nugget.
   - Key qualifiers mismatch (time, modality, scope, quantities, or conditions).

Rules:
- Use only the passage; ignore external knowledge.
- Base judgments on verifiable snippets from the passage; quote the minimal necessary phrase(s).
- If information is missing or contradicted → not_support.
- If some relevant evidence exists but is incomplete → partial_support.
- Paraphrases count as support if the meaning clearly aligns.

Output format (two-part):
1) Reasoning: Wrap ALL nugget-level reasoning in a single block delimited by <reasoning> and </reasoning>. Inside the block, for each nugget k = 1..{num_nuggets}, output a standardized multi-line item with the four fields in order: nugget, snippet, rationale, decision. Use the following format for every k:
   <reasoning>
   k.
     nugget="<nugget_k>"
     snippet="<verbatim phrase(s) from passage>"
     rationale="<free-form reasoning; one or more sentences/clauses>"
     decision="<support|partial_support|not_support>"
   ...
   </reasoning>
   - The rationale may be any length/format (short clause, sentence(s), or bullet-like clauses separated by ';' or new lines). Do NOT include square brackets [] anywhere in reasoning. Do NOT output any other tags outside <reasoning> ... </reasoning>.
2) Final answer: The LAST part of your response must be the labels in the requested format.
   - {format_instruction}
   - No extra text after the labels.
   - Count must equal {num_nuggets}."""

    def _get_short_cot_user(self, format_type: str) -> str:
        """User prompt for short chain-of-thought mode"""
        return """Search Query: {query}
Passage: {passage}
Nuggets ({num_nuggets}): {nugget_list}

Begin reasoning and final labels:"""
    
    def _get_optimized_system(self, format_type: str) -> str:
        """System prompt for optimized decision-tree based mode"""
        return """You are NuggetMatchJudge.

Task:
Given a search query, a passage, and {num_nuggets} nuggets, assign one label to each nugget: "support", "partial_support", or "not_support".

Core Principle:
Your judgment must be based EXCLUSIVELY on the provided passage. Do not use any external knowledge.

Label Definitions & Decision Process:
Please follow this decision framework for each nugget:

1. Check for Contradiction → "not_support"
   - Does the passage explicitly state the opposite of the nugget?
   - If yes, label "not_support".

2. Check for Full Support → "support"  
   - Are ALL essential facts of the nugget explicitly and unambiguously stated in the passage?
   - Essential facts include: subjects, actions, key quantities, dates, conditions, and qualifiers
   - Do all qualifiers (e.g., "always", "some", "may") match perfectly?
   - If yes, label "support".

3. Check for Partial Support → "partial_support"
   - Does the passage support at least one essential fact, but another essential fact is missing, hedged (e.g., "may", "suggests"), or stated ambiguously?
   - Does verifying the nugget require only a minor, safe inference (e.g., treating clear paraphrases like "reached the summit" as equivalent to "climbed the mountain")?
   - If yes, label "partial_support".
   - Safe inference example: Passage says "turnover of $10 million", nugget says "revenue of $10 million"
   - Unsafe inference example: Passage says "CEO bought a new car", nugget says "company is doing well financially"

4. Default → "not_support"
   - If none of the above conditions are met (information entirely absent or only topically related), label "not_support".

Output Format (two parts):
1) Reasoning: Place ALL your reasoning analysis inside <reasoning> ... </reasoning> tags. For each nugget, freely express your thought process, including:
   - Restate the nugget to ensure understanding
   - Quote or paraphrase relevant parts from the passage
   - Analyze the relationship and support level
   - Reach a conclusion (support/partial_support/not_support)
   Use any format that helps you think clearly - paragraphs, bullet points, or numbered lists.

2) Final Answer: After the </reasoning> tag, provide the final labels in the requested format.
   - {format_instruction}
   - No extra text after the labels.
   - Before submitting the Final Answer, confirm 3 points: (1) Order matches nugget serial numbers; (2) No repeated labels for any nugget; (3) Number of labels = {num_nuggets}. Only submit if all 3 points are satisfied.
   
   """
    
    def _get_optimized_user(self, format_type: str) -> str:
        """User prompt for optimized mode"""
        return """Search Query: {query}
Passage: {passage}
Nuggets ({num_nuggets}): {nugget_list}

Please provide your detailed reasoning in <reasoning> tags, then collect the final result for each nugget from the reasoning section and list them in order :"""

    def _get_optimizednum_system(self, format_type: str) -> str:
        """System prompt for optimized decision-tree based mode"""
        return """You are NuggetMatchJudge.

Task:
Given a search query, a passage, and {num_nuggets} nuggets, assign one label to each nugget: "support", "partial_support", or "not_support".

Core Principle:
Your judgment must be based EXCLUSIVELY on the provided passage. Do not use any external knowledge.

Label Definitions & Decision Process:
Please follow this decision framework for each nugget:

1. Check for Contradiction → "not_support"
   - Does the passage explicitly state the opposite of the nugget?
   - If yes, label "not_support".

2. Check for Full Support → "support"  
   - Are ALL essential facts of the nugget explicitly and unambiguously stated in the passage?
   - Essential facts include: subjects, actions, key quantities, dates, conditions, and qualifiers
   - Do all qualifiers (e.g., "always", "some", "may") match perfectly?
   - If yes, label "support".

3. Check for Partial Support → "partial_support"
   - Does the passage support at least one essential fact, but another essential fact is missing, hedged (e.g., "may", "suggests"), or stated ambiguously?
   - Does verifying the nugget require only a minor, safe inference (e.g., treating clear paraphrases like "reached the summit" as equivalent to "climbed the mountain")?
   - If yes, label "partial_support".
   - Safe inference example: Passage says "turnover of $10 million", nugget says "revenue of $10 million"
   - Unsafe inference example: Passage says "CEO bought a new car", nugget says "company is doing well financially"

4. Default → "not_support"
   - If none of the above conditions are met (information entirely absent or only topically related), label "not_support".

Output Format (two parts):
1) Reasoning: Place ALL your reasoning analysis inside <reasoning> ... </reasoning> tags. For each nugget, freely express your thought process, including:
   - Restate the nugget to ensure understanding
   - Quote or paraphrase relevant parts from the passage
   - Analyze the relationship and support level
   - Reach a conclusion (support/partial_support/not_support)
   Use numbered lists format .

2) Final Answer: After the </reasoning> tag, provide the final labels in the requested format.
   - 'Respond with a numbered list, one label for each nugget:\n'
                '1. support\n'
                '2. partial_support\n'
                '3. not_support\n'
                'Example:\n1. support\n2. not_support\n3. partial_support'
   - No extra text after the labels.
   - Before submitting the Final Answer, confirm 3 points:
    (1) Check the numbering order: Does it start from 1 and increase consecutively (e.g., 1→2→3→4→5→6)? If there is an out-of-order sequence such as "3→4→1", correct it immediately;
    (2) Check the uniqueness of numbers: Does each number (1-6) appear only once? If there are duplicates (e.g., two entries of "3. Nugget 3"), delete the duplicate entries immediately;
    (3) Check the completeness of quantity: Does the number of labels equal {num_nuggets} (e.g., 6)? If one label is missing, supplement the missing numbered label immediately.
   """
    
    def _get_optimizednum_user(self, format_type: str) -> str:
        """User prompt for optimized mode"""
        return """Search Query: {query}
Passage: {passage}
Nuggets ({num_nuggets}): {nugget_list}
Please provide your detailed reasoning in <reasoning> tags, then collect the final result for each nugget from the reasoning section and list them in order :"""


    def _get_optimizedhot_system(self, format_type: str) -> str:
        """System prompt for optimized decision-tree based mode"""
        return """You are NuggetMatchJudge.

Task:
Given a search query, a passage, and {num_nuggets} nuggets, assign one label to each nugget: "support" or "not_support".

Core Principle:
Your judgment must be based EXCLUSIVELY on the provided passage. Do not use any external knowledge.

Label Definitions & Decision Process:
Please follow this decision framework for each nugget:

1. Check for Contradiction → "not_support"
   - Does the passage explicitly state the opposite of the nugget?
   - If yes, label "not_support".

2. Check for Full Support → "support"  
   - Are ALL essential facts of the nugget explicitly and unambiguously stated in the passage?
   - Essential facts include: subjects, actions, key quantities, dates, conditions, and qualifiers
   - Do all qualifiers (e.g., "always", "some", "may") match perfectly?
   - If yes, label "support".

3. Default → "not_support"
   - If none of the above conditions are met (information entirely absent or only topically related), label "not_support".

Output Format (two parts):
1) Reasoning: Place ALL your reasoning analysis inside <reasoning> ... </reasoning> tags. For each nugget, freely express your thought process, including:
   - Restate the nugget to ensure understanding
   - Quote or paraphrase relevant parts from the passage
   - Analyze the relationship and support level
   - Reach a conclusion (support/not_support)
   Use any format that helps you think clearly - paragraphs, bullet points, or numbered lists.

2) Final Answer: After the </reasoning> tag, provide the final labels in the requested format.
   - {format_instruction}
   - No extra text after the labels.
   - Before submitting the Final Answer, confirm 3 points: (1) Order matches nugget serial numbers; (2) No repeated labels for any nugget; (3) Number of labels = {num_nuggets}. Only submit if all 3 points are satisfied.

   """

    def _get_optimizedhot_user(self, format_type: str) -> str:
        """User prompt for optimized mode"""
        return """Search Query: {query}
Passage: {passage}
Nuggets ({num_nuggets}): {nugget_list}

Please provide your detailed reasoning in <reasoning> tags, then collect the final result for each nugget from the reasoning section and list them in order :"""


    def build_template_prompt(self, prompt_type: str, query: str, passage: str, 
                            nuggets: List[str], format_type: str = "adaptive") -> List[Dict[str, str]]:
        """
        Build prompt messages using template system with format support
        
        Args:
            prompt_type: Prompt type (no_reasoning|short_cot|long_cot|optimized)
            query: Search query
            passage: Text passage
            nuggets: List of nugget texts
            format_type: Output format type
            
        Returns:
            List of message dictionaries for the API call
        """
        # Format nugget list for display
        nugget_list = "\n".join([f"{i+1}. {nugget}" for i, nugget in enumerate(nuggets)])
        
        # Get format instruction
        format_instruction = self._get_format_instruction(format_type)
        
        # Build messages based on prompt type
        if prompt_type == "no_reasoning":
            system_content = self._get_no_reasoning_system(format_type).format(
                num_nuggets=len(nuggets),
                format_instruction=format_instruction
            )
            user_content = self._get_no_reasoning_user(format_type).format(
                query=query,
                passage=passage,
                nugget_list=nugget_list,
                format_instruction=format_instruction
            )
        elif prompt_type == "short_cot":
            system_content = self._get_short_cot_system(format_type).format(
                num_nuggets=len(nuggets),
                format_instruction=format_instruction
            )
            user_content = self._get_short_cot_user(format_type).format(
                query=query,
                passage=passage,
                nugget_list=nugget_list,
                num_nuggets=len(nuggets)
            )
        elif prompt_type == "long_cot":
            system_content = self._get_long_cot_system(format_type).format(
                num_nuggets=len(nuggets),
                format_instruction=format_instruction
            )
            user_content = self._get_long_cot_user(format_type).format(
                query=query,
                passage=passage,
                nugget_list=nugget_list,
                num_nuggets=len(nuggets)
            )
        elif prompt_type == "optimized":
            system_content = self._get_optimized_system(format_type).format(
                num_nuggets=len(nuggets),
                format_instruction=format_instruction
            )
            user_content = self._get_optimized_user(format_type).format(
                query=query,
                passage=passage,
                nugget_list=nugget_list,
                num_nuggets=len(nuggets)
            )
        elif prompt_type == "optimizednum":
            system_content = self._get_optimizednum_system(format_type).format(
                num_nuggets=len(nuggets),
                format_instruction=format_instruction
            )
            user_content = self._get_optimizednum_user(format_type).format(
                query=query,
                passage=passage,
                nugget_list=nugget_list,
                num_nuggets=len(nuggets)
            )
        elif prompt_type == "optimizedhot":
            system_content = self._get_optimizedhot_system(format_type).format(
                num_nuggets=len(nuggets),
                format_instruction=format_instruction
            )
            user_content = self._get_optimizedhot_user(format_type).format(
                query=query,
                passage=passage,
                nugget_list=nugget_list,
                num_nuggets=len(nuggets)
            )
        else:
            raise ValueError(f"Invalid prompt_type: {prompt_type}")
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def is_template_prompt_type(self, prompt_type: str) -> bool:
        """Check if the given prompt type is a template-based type"""
        return prompt_type in ["no_reasoning", "short_cot", "long_cot", "optimized","optimizednum","optimizedhot"]

    def get_available_prompt_types(self) -> List[str]:
        """Get list of available prompt types"""
        return ["legacy", "no_reasoning", "short_cot", "long_cot", "optimized","optimizednum","optimizedhot"]


# Global instance for easy access
prompt_template_manager = PromptTemplateManager()


def build_template_prompt(prompt_type: str, query: str, passage: str, nuggets: List[str], 
                         format_type: str = "adaptive") -> List[Dict[str, str]]:
    """
    Convenience function to build template prompts
    
    Args:
        prompt_type: Prompt type (no_reasoning|short_cot|long_cot|optimized|optimizednum)
        query: Search query
        passage: Text passage  
        nuggets: List of nugget texts
        format_type: Output format type
        
    Returns:
        List of message dictionaries
    """
    return prompt_template_manager.build_template_prompt(prompt_type, query, passage, nuggets, format_type)