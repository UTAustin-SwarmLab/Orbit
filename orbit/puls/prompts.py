def find_prompt(prompt):
    full_prompt = f"""
You are an intelligent agent designed to extract structured representations from video question prompts. You will operate in two stages: (1) proposition extraction and (2) TL specification generation.

Stage 1: Proposition Extraction

Given an input question about a video, extract the atomic propositions that describe the underlying events or facts explicity referenced in the question. These propositions should describe object-action or object-object relationships stated in the question — avoid making assumptions or inferring any additional events. Avoid TL keywords such as 'and', 'or', 'not', 'until'.
Do not include ambiguous propositions that lack specificity. For instance, phrases like "guy does something" are ambiguous and should be omitted. Instead, focus on concrete actions or relationships. For example, given the prompt "In a bustling park, a child kicks a ball. What happens when the ball hits the bench?", the correct propositions are ["child kicks ball", "ball hits bench"]. If a proposition is getting too complex, try to simplify it by extracting the main, important points---for example, instead of "subject tightens the valve nut of the wheel on the floor with his right hand", try to say "subject tightens nut".

Stage 2: TL Specification Generation

Using only the list of the propositions extracted in Stage 1, generate a single Temporal Logic (TL) specification that catpures the sequence of logical structure implied by the question. 

Rules:
- The formula must use each proposition **exactly once**
- Use only the TL operators: `AND`, `OR`, `NOT`, `UNTIL`
- Do **not** infer new events or rephrase propositions.
- The formula should reflect the temporal or logical relationships between the propositions under which the question would be understandable.

**Examples**

Example 1: "In a sunny meadow, a child plays with a kite and runs around. What does the child do after falling?"
Output:
{{
  "proposition": ["child plays with kite", "child runs around", "child falls"],
  "specification": "(child plays with kite AND child runs around) UNTIL child falls"
}}

Example 2: "In a dimly lit room, two robots stand silently. What happens when the red robot starts blinking or the green robot does not turn off?"
Output:
{{
  "proposition": ["robots stand silently", "red robot blinks", "green robot turns off"],
  "specification": "robots stand silently UNTIL (red robot blinks OR NOT green robot turns off)"
}}

Example 3: "Inside a cave, a man holds a lantern. What happens when the man sees the dragon?"
Output:
{{
  "proposition": ["man holds lantern", "man sees dragon"],
  "specification": "man holds lantern UNTIL man sees dragon"
}}

Example 4: "How did the girl feel before turning on the computer?"
Output:
{{
  "proposition": ["girl turns on computer"],
  "specification": "(girl turns on computer)"
}}

**Now process the following prompt:**
Input:
{{
  "prompt": "{prompt}"
}}

Expected Output (only output the following JSON structure — nothing else):
{{
  "proposition": [...],
  "specification": "..."
}}
"""
    return full_prompt

