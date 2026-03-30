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


SYSTEM_TEXT_INTERPRET_V2 = (
    "You convert one natural-language UI automation command into one strict action spec. "
    "Return exactly one valid JSON object and nothing else. "
    "Preserve only attributes that the user explicitly requested or clearly implied. "
    "Keep visible text exactly as written. "
    "Interpret the command into the best strict target description you can. "
    "Do not require proof that the target exists in the screenshot at this stage. "
    "Use status='ambiguous' only when the user's intent itself is unclear or underspecified."
)


SYSTEM_TEXT_GROUND_V2 = (
    "You analyze exactly one screenshot of a website or application UI. "
    "Return exactly one valid JSON object and nothing else. "
    "Use only information visibly present in the screenshot. "
    "The provided action spec is the source of truth. "
    "If there is no exact visible match, return not_found. "
    "If several candidates still match, return ambiguous."
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


def build_interpret_prompt_v2(
    mode: str,
    question: str,
    requested_input_length: int | None = None,
) -> str:
    length_hint = ""
    if requested_input_length is not None:
        length_hint = (
            "- if the user requests random text and gives only a length hint, "
            f"use random_length={requested_input_length}\n"
        )

    if mode in {"yes_no", "point"}:
        return (
            "Return exactly one JSON object.\n"
            "Required keys:\n"
            "- status: ok or ambiguous\n"
            "- target_description: string or null\n"
            "- source_description: null\n"
            "- destination_description: null\n"
            "- input_text: empty string\n"
            "- generate_random_text: false\n"
            "- random_length: null\n"
            "- comment: string\n"
            "Rules:\n"
            "- do not decide whether the target exists or is unique at this stage\n"
            "- only convert the user command into the best strict description of the intended target\n"
            "- if the user gives a plausible target description, return status=ok with the best strict target_description\n"
            "- preserve only attributes that are explicit or clearly implied\n"
            "- preserve visible text exactly as written\n"
            "- do not invent missing color, type, state, label, or position\n"
            "- location hints like top left, bottom right, near, below, above, left, right are valid constraints and should be preserved\n"
            "- only use ambiguous if the command itself does not identify what object to check or point to\n"
            f"User command: {question}"
        )

    if mode == "input":
        return (
            "Return exactly one JSON object.\n"
            "Required keys:\n"
            "- status: ok or ambiguous\n"
            "- target_description: string or null\n"
            "- source_description: null\n"
            "- destination_description: null\n"
            "- input_text: string\n"
            "- generate_random_text: boolean\n"
            "- random_length: integer or null\n"
            "- comment: string\n"
            "Rules:\n"
            "- do not decide whether the target exists or is unique at this stage\n"
            "- only convert the user command into the best strict description of the intended editable field\n"
            "- if the user gives a plausible editable field description, return status=ok with the best strict target_description\n"
            "- preserve visible labels, placeholders, or exact text exactly as written\n"
            "- the target must describe the editable field itself, not only a nearby label\n"
            "- relative field descriptions like below Brand label are valid and should be preserved\n"
            "- only use ambiguous if the command itself does not identify which field to use\n"
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
            "- target_description: null\n"
            "- source_description: string or null\n"
            "- destination_description: string or null\n"
            "- input_text: empty string\n"
            "- generate_random_text: false\n"
            "- random_length: null\n"
            "- comment: string\n"
            "Rules:\n"
            "- do not decide whether the source or destination exists or is unique at this stage\n"
            "- only convert the user command into the best strict source and destination descriptions\n"
            "- if the user gives a plausible source and destination, return status=ok with the best strict descriptions\n"
            "- both source_description and destination_description must be present for status=ok\n"
            "- preserve explicit text exactly as written\n"
            "- do not invent missing source or destination details\n"
            "- relative destinations like middle of the slider track are valid and should be preserved\n"
            "- only use ambiguous if the command itself leaves source or destination unclear\n"
            f"User command: {question}"
        )

    if mode == "value":
        return (
            "Return exactly one JSON object.\n"
            "Required keys:\n"
            "- status: ok or ambiguous\n"
            "- target_description: string or null\n"
            "- source_description: null\n"
            "- destination_description: null\n"
            "- input_text: empty string\n"
            "- generate_random_text: false\n"
            "- random_length: null\n"
            "- comment: string\n"
            "Rules:\n"
            "- do not decide whether the target exists or is unique at this stage\n"
            "- only convert the user command into the best strict description of the single value-bearing target\n"
            "- if the user asks how many, what number, what count, or what value, identify the UI element that displays that value\n"
            "- if the user gives a plausible target, return status=ok with the best strict target_description\n"
            "- preserve visible text and labels exactly as written\n"
            "- preserve location hints such as top right, top left, below, above, near\n"
            "- the user does not need to provide the value itself; they only need to identify where it should be read from\n"
            "- only use ambiguous if the command itself does not identify which displayed value should be read\n"
            "Examples:\n"
            "- how many items shown in shopping cart in top right corner? -> shopping cart item count badge in the top right corner\n"
            "- what number is shown on the cart badge? -> cart badge number\n"
            f"User command: {question}"
        )

    if mode == "multi_value":
        return (
            "Return exactly one JSON object.\n"
            "Required keys:\n"
            "- status: ok or ambiguous\n"
            "- target_description: string or null\n"
            "- source_description: null\n"
            "- destination_description: null\n"
            "- input_text: empty string\n"
            "- generate_random_text: false\n"
            "- random_length: null\n"
            "- comment: string\n"
            "Rules:\n"
            "- do not decide whether the target exists or is unique at this stage\n"
            "- only convert the user command into the best strict description of the list or group whose visible items should be read\n"
            "- if the user gives a plausible list target, return status=ok with the best strict target_description\n"
            "- preserve visible text and labels exactly as written\n"
            "- only use ambiguous if the command itself does not identify which list or group should be read\n"
            f"User command: {question}"
        )

    raise ValueError(f"Unsupported mode: {mode}")


def build_ground_prompt_v2(mode: str, task_spec: dict[str, object], coord_max: int) -> str:
    import json

    if mode == "yes_no":
        return (
            "Return exactly one JSON object.\n"
            "Required keys:\n"
            "- status: found, not_found, or ambiguous\n"
            "- comment: string; brief reason why the target was found, not found, or ambiguous\n"
            "Rules:\n"
            "- use the task spec as the source of truth\n"
            "- the target must satisfy all attributes in the target_description\n"
            "- if any required attribute is missing or mismatched, return not_found\n"
            "- if several candidates still match, return ambiguous\n"
            f"Task spec: {json.dumps(task_spec, ensure_ascii=False)}"
        )

    if mode in {"point", "input"}:
        target_kind = "editable input field" if mode == "input" else "target element"
        return (
            "Return exactly one JSON object.\n"
            "Required keys:\n"
            "- status: found, not_found, or ambiguous\n"
            f"- x: integer from 0 to {coord_max}, or null\n"
            f"- y: integer from 0 to {coord_max}, or null\n"
            "- comment: string; brief reason why the coordinates were returned or why they are null\n"
            "Rules:\n"
            "- use the task spec as the source of truth\n"
            "- return normalized coordinates across the full image\n"
            f"- return the center of the {target_kind} itself\n"
            "- do not return a nearby label, icon, text, or container\n"
            "- if any required attribute is missing or mismatched, return not_found with nulls\n"
            "- if several candidates still match, return ambiguous with nulls\n"
            "- never return coordinates for a merely similar element\n"
            f"Task spec: {json.dumps(task_spec, ensure_ascii=False)}"
        )

    if mode == "drag":
        return (
            "Return exactly one JSON object.\n"
            "Required keys:\n"
            "- status: found, not_found, or ambiguous\n"
            f"- x, y, x2, y2: integers from 0 to {coord_max}, or null\n"
            "- comment: string; brief reason why the drag coordinates were returned or why they are null\n"
            "Rules:\n"
            "- use the task spec as the source of truth\n"
            "- x,y is the center of the draggable source element\n"
            "- x2,y2 is the center of the destination drop point or destination element\n"
            "- both endpoints must match exactly\n"
            "- if either endpoint is missing, return not_found with nulls\n"
            "- if either endpoint is ambiguous, return ambiguous with nulls\n"
            "- never use a similar element as a fallback\n"
            f"Task spec: {json.dumps(task_spec, ensure_ascii=False)}"
        )

    if mode == "value":
        return (
            "Return exactly one JSON object.\n"
            "Required keys:\n"
            "- status: found, not_found, or ambiguous\n"
            "- answer: string; the exact visible value, or empty string if not_found or ambiguous\n"
            "- comment: string; brief reason why the value was returned or why it is empty\n"
            "Rules:\n"
            "- use the task spec as the source of truth\n"
            "- return exactly one visible value from the target element or region\n"
            "- preserve visible text exactly as written\n"
            "- if the required target is missing or mismatched, return not_found with answer=''\n"
            "- if several candidates still match, return ambiguous with answer=''\n"
            "- do not guess hidden, cropped, or inferred values\n"
            f"Task spec: {json.dumps(task_spec, ensure_ascii=False)}"
        )

    if mode == "multi_value":
        return (
            "Return exactly one JSON object.\n"
            "Required keys:\n"
            "- status: found, not_found, or ambiguous\n"
            "- answer: array of strings; the visible items in order, or [] if not_found or ambiguous\n"
            "- comment: string; brief reason why the list was returned or why it is empty\n"
            "Rules:\n"
            "- use the task spec as the source of truth\n"
            "- return only the visible items that belong to the requested list or group\n"
            "- preserve visible text exactly as written\n"
            "- return items in visual order from top to bottom, or left to right when appropriate\n"
            "- if the required target is missing or mismatched, return not_found with answer=[]\n"
            "- if several candidate lists or groups still match, return ambiguous with answer=[]\n"
            "- do not guess hidden, cropped, or inferred items\n"
            f"Task spec: {json.dumps(task_spec, ensure_ascii=False)}"
        )

    raise ValueError(f"Unsupported mode: {mode}")
