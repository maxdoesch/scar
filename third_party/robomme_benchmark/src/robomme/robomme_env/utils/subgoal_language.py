from ...logging_utils import logger


def get_subgoal_with_index(idx, template, **kwargs):
    if idx == 0:
        subgoal = template.format(idx="first", **kwargs)
    elif idx == 1:
        subgoal = template.format(idx="second", **kwargs)
    elif idx == 2:
        subgoal = template.format(idx="third", **kwargs)
    elif idx == 3:
        subgoal = template.format(idx="fourth", **kwargs)
    elif idx == 4:
        subgoal = template.format(idx="fifth", **kwargs)
    elif idx == 5:
        subgoal = template.format(idx="sixth", **kwargs)
    elif idx == 6:
        subgoal = template.format(idx="seventh", **kwargs)
    elif idx == 7:
        subgoal = template.format(idx="eighth", **kwargs)
    elif idx == 8:
        subgoal = template.format(idx="ninth", **kwargs)
    elif idx == 9:
        subgoal = template.format(idx="tenth", **kwargs)
    else:
        raise ValueError(f"Invalid index: {idx}")
    return subgoal


_CARDINAL_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def get_subgoal_with_count(count, template, **kwargs):
    """Render a subgoal template substituting {count} and {plural}, e.g.
    get_subgoal_with_count(2, "...{count} time{plural}") -> "...two times"."""
    if count not in _CARDINAL_WORDS:
        raise ValueError(f"Invalid count: {count}")
    plural = "" if count == 1 else "s"
    return template.format(count=_CARDINAL_WORDS[count], plural=plural, **kwargs)


if __name__ == "__main__":
    logger.debug(
        get_subgoal_with_index(0, "pick up the {idx} {color} cube", color="red")
    )
