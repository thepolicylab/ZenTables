"""
Provides ways to change global options
"""

from dataclasses import dataclass


@dataclass
class OptionsWrapper:
    """A wrapper class around a dict to provide global options functionalities."""

    font_family: str = "Arial, Helvetica, sans-serif"
    font_size: str = "11pt"
    hide_index_names: bool = True
    show_copy_button: bool = True


_options = OptionsWrapper()


def set_options(**kwargs):
    """Utility function to set package-wide options.

    Args:
        kwargs: pass into the function the option name and value to be set.

    Raises:
        KeyError: if the option passed is not a valid option.

    Examples:
        import zentables as zen
        zen.set_options(option1=value1, option2=value2)
    """
    for opt, val in kwargs.items():
        if hasattr(_options, opt):
            setattr(_options, opt, val)
        else:
            raise KeyError(f"Invalid option: {opt}")
