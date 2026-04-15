SYSTEM_TEXT = (
    "You analyze exactly one screenshot of a website or application UI. "
    "Return exactly one valid JSON object and nothing else. "
    "Use only information visibly present in the screenshot. "
    "The provided action spec is the source of truth. "
    "If there is no exact visible match, return not_found. "
    "If several candidates still match, return ambiguous."
)

NORMALIZED_COORD_MAX = 1000

PROMPT_HEADER = (
    "You receive one screenshot and one free-form user request.\n"
    "The required JSON schema and task rules are defined below.\n"
    "Interpret the user's wording only as much as needed to complete that task on the screenshot.\n"
    "Use only what is visible in the screenshot.\n"
    "Do not invent missing attributes, hidden values, or off-screen content.\n"
    "Return exactly one valid JSON object and nothing else.\n"
    "User request: {question}\n"
)

prom = {
    "yes_no": (
        PROMPT_HEADER +
        "Required keys:\n"
        "- status: found, not_found, or ambiguous\n"
        "- comment: string; brief reason why the target was found, not found, or ambiguous\n"
        "Rules:\n"
        "- decide whether the screenshot contains the target implied by the user request\n"
        "- preserve explicit text, color, position, relation, and state constraints from the user request\n"
        "- if there is no exact visible match, return not_found\n"
        "- if several candidates still match, return ambiguous\n"
        "- do not relax the request to a merely similar element"
    ),
    "point": (
        PROMPT_HEADER +
        "Required keys:\n"
        "- status: found, not_found, or ambiguous\n"
        f"- x: integer from 0 to {NORMALIZED_COORD_MAX}, or null\n"
        f"- y: integer from 0 to {NORMALIZED_COORD_MAX}, or null\n"
        "- comment: string; brief reason why the coordinates were returned or why they are null\n"
        "Rules:\n"
        "- locate the single target element implied by the user request\n"
        "- preserve explicit text, color, position, relation, and state constraints from the user request\n"
        "- return normalized coordinates across the full image\n"
        "- return the center of the target element itself\n"
        "- do not return a nearby label, icon, text, or container\n"
        "- if there is no exact visible match, return not_found with nulls\n"
        "- if several candidates still match, return ambiguous with nulls\n"
        "- do not return coordinates for a merely similar element"
    ),
    "input": (
        PROMPT_HEADER +
        "Required keys:\n"
        "- status: found, not_found, or ambiguous\n"
        f"- x: integer from 0 to {NORMALIZED_COORD_MAX}, or null\n"
        f"- y: integer from 0 to {NORMALIZED_COORD_MAX}, or null\n"
        "- text: string; the exact text that should be typed based on the user request\n"
        "- comment: string; brief reason why the coordinates and text were returned\n"
        "Rules:\n"
        "- identify the editable field the user wants to type into\n"
        "- determine the text to type directly from the free-form user request\n"
        "- if the request asks for generated or random text, choose it yourself and return the exact chosen text\n"
        "- if the request does not specify any text to type, return text as an empty string\n"
        "- return normalized coordinates across the full image\n"
        "- return the center of the editable input field itself\n"
        "- do not return a nearby label, icon, text, or container\n"
        "- if there is no exact visible match, return not_found with nulls\n"
        "- if several candidates still match, return ambiguous with nulls\n"
        "- do not treat the requested text value itself as proof that the field is present"
    ),
    "drag": (
        PROMPT_HEADER +
        "Required keys:\n"
        "- status: found, not_found, or ambiguous\n"
        f"- x, y, x2, y2: integers from 0 to {NORMALIZED_COORD_MAX}, or null\n"
        "- comment: string; brief reason why the drag coordinates were returned or why they are null\n"
        "Rules:\n"
        "- infer the draggable source and destination directly from the free-form user request\n"
        "- x,y is the center of the draggable source element\n"
        "- x2,y2 is the center of the destination drop point or destination element\n"
        "- preserve explicit text, color, position, relation, and state constraints from the user request\n"
        "- both endpoints must match the request exactly\n"
        "- if either endpoint is missing, return not_found with nulls\n"
        "- if either endpoint is ambiguous, return ambiguous with nulls\n"
        "- never use a merely similar source or destination as a fallback\n"
        "- do not substitute a generic center point when the destination is underspecified"
    ),
    "value": (
        PROMPT_HEADER +
        "Required keys:\n"
        "- status: found, not_found, or ambiguous\n"
        "- answer: string; the exact visible value, or empty string if not_found or ambiguous\n"
        "- comment: string; brief reason why the value was returned or why it is empty\n"
        "Rules:\n"
        "- identify from the free-form request which single visible value should be read\n"
        "- return exactly one visible value from the requested element or region\n"
        "- preserve visible text exactly as written\n"
        "- if the required target is missing or mismatched, return not_found with answer=''\n"
        "- if several candidates still match, return ambiguous with answer=''\n"
        "- do not guess hidden, cropped, or inferred values\n"
        "- do not return labels or surrounding text unless they are the value itself"
    ),
    "multi_value": (
        PROMPT_HEADER +
        "Required keys:\n"
        "- status: found, not_found, or ambiguous\n"
        "- answer: array of strings; the visible items in order, or [] if not_found or ambiguous\n"
        "- comment: string; brief reason why the list was returned or why it is empty\n"
        "Rules:\n"
        "- identify from the free-form request which list, group, or collection should be read\n"
        "- return only the visible items that belong to the requested list or group\n"
        "- preserve visible text exactly as written\n"
        "- return items in visual order from top to bottom, or left to right when appropriate\n"
        "- if the required target is missing or mismatched, return not_found with answer=[]\n"
        "- if several candidate lists or groups still match, return ambiguous with answer=[]\n"
        "- do not guess hidden, cropped, or inferred items\n"
        "- do not merge items from different groups or sections"
    ),
}
