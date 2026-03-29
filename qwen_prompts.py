SYSTEM_TEXT_PARSE = (
    "You convert a user's natural-language UI command into a strict internal task. "
    "Return exactly one valid JSON object and nothing else. "
    "Preserve only attributes that the user explicitly requested or clearly implied. "
    "Keep visible text snippets exactly as written, including letters, digits, and punctuation. "
    "Do not invent missing color, type, state, label, placeholder, or destination details. "
    "If the command is ambiguous, return status='ambiguous' instead of guessing."
)


SYSTEM_TEXT_VISION = (
    "You analyze exactly one screenshot of a website or web application. "
    "Return exactly one valid JSON object and nothing else. "
    "Use only information visibly present in the screenshot. "
    "The provided internal task is the source of truth. "
    "Every non-null attribute in the internal task must match. "
    "If there is no exact visible match, return not_found. "
    "If several candidates still match and the target is not uniquely determined, return ambiguous."
)


def build_parse_prompt(
    mode: str,
    question: str,
    requested_input_length: int | None = None,
) -> str:
    if mode in {"yes_no", "point"}:
        return (
            "Return exactly one JSON object.\n"
            "Required keys:\n"
            "- status: ok or ambiguous\n"
            "- target_description: string or null; a short strict internal description for screenshot grounding\n"
            "- comment: string; brief reason for the parsed result\n"
            "Rules:\n"
            "- preserve only attributes that are explicit or clearly implied\n"
            "- preserve visible text exactly as written\n"
            "- do not invent missing color, type, state, label, or position\n"
            "- if the user uses a vague location reference without enough context, use ambiguous\n"
            "- target_description should be one short phrase, not a full sentence\n"
            f"Mode: {mode}\n"
            f"User command: {question}"
        )

    if mode == "input":
        length_hint = ""
        if requested_input_length is not None:
            length_hint = (
                "- if random text is requested and length is unclear, "
                f"use random_length={requested_input_length}\\n"
            )

        return (
            "Return exactly one JSON object.\n"
            "Required keys:\n"
            "- status: ok or ambiguous\n"
            "- target_description: string or null; strict description of the editable input field\n"
            "- input_text: string; literal text to type, or empty string if not explicitly provided\n"
            "- generate_random_text: true only if the user explicitly asks for random text\n"
            "- random_length: integer length for random text, or null\n"
            "- comment: string; brief reason for the parsed result\n"
            "Rules:\n"
            "- preserve visible labels, placeholders, or exact text exactly as written\n"
            "- do not use a label as the target itself; the target is the editable field\n"
            "- if the command does not identify a unique target field, use ambiguous\n"
            "- if the command asks for random text, keep input_text empty and set generate_random_text=true\n"
            "- if the command provides literal text to enter, copy it exactly into input_text\n"
            f"{length_hint}"
            f"User command: {question}"
        )

    if mode == "drag":
        return (
            "Return exactly one JSON object.\n"
            "Required keys:\n"
            "- status: ok or ambiguous\n"
            "- source_description: string or null; strict description of what should be dragged\n"
            "- destination_description: string or null; strict description of where it should be dropped\n"
            "- comment: string; brief reason for the parsed result\n"
            "Rules:\n"
            "- both source_description and destination_description must be present for status=ok\n"
            "- preserve explicit text exactly as written\n"
            "- do not invent missing source or destination details\n"
            "- if the command does not identify both endpoints clearly enough, use ambiguous\n"
            f"User command: {question}"
        )

    raise ValueError("Unsupported mode")


def build_exists_prompt(target_description: str) -> str:
    return (
        "Return exactly one JSON object.\n"
        "Required keys:\n"
        "- status: found, not_found, or ambiguous\n"
        "- comment: string; brief reason why the target was found, not found, or ambiguous\n"
        "Rules:\n"
        "- use the internal task as the source of truth\n"
        "- the target must satisfy all attributes in the description\n"
        "- if any required attribute is missing or mismatched, return not_found\n"
        "- if several candidates still match, return ambiguous\n"
        f"Internal task: {target_description}"
    )


def build_point_prompt(target_description: str, coord_max: int) -> str:
    return (
        "Return exactly one JSON object.\n"
        "Required keys:\n"
        "- status: found, not_found, or ambiguous\n"
        f"- x: integer from 0 to {coord_max}, or null\n"
        f"- y: integer from 0 to {coord_max}, or null\n"
        "- comment: string; brief reason why the coordinates were returned or why they are null\n"
        "Rules:\n"
        "- return normalized coordinates across the full image\n"
        "- return the center of the target element itself\n"
        "- for buttons, use the center of the clickable button rectangle\n"
        "- for input fields, use the center of the editable text box\n"
        "- do not return a nearby label, icon, text, or container\n"
        "- if any required attribute is missing or mismatched, return not_found with nulls\n"
        "- if several candidates still match, return ambiguous with nulls\n"
        "- never return coordinates for a merely similar element\n"
        f"Internal task: {target_description}"
    )


def build_drag_prompt(
    source_description: str,
    destination_description: str,
    coord_max: int,
) -> str:
    return (
        "Return exactly one JSON object.\n"
        "Required keys:\n"
        "- status: found, not_found, or ambiguous\n"
        f"- x, y, x2, y2: integers from 0 to {coord_max}, or null\n"
        "- comment: string; brief reason why the drag coordinates were returned or why they are null\n"
        "Rules:\n"
        "- x,y is the center of the draggable source element\n"
        "- x2,y2 is the center of the destination drop point or destination element\n"
        "- both endpoints must match exactly\n"
        "- if either endpoint is missing, return not_found with nulls\n"
        "- if either endpoint is ambiguous, return ambiguous with nulls\n"
        "- never use a similar element as a fallback\n"
        f"Source task: {source_description}\n"
        f"Destination task: {destination_description}"
    )
