"""
Useful enum implementations
"""
import json
from string import Template
from typing import List

from aenum import MultiValueEnum


class BaseEOGrowEnum(MultiValueEnum):
    """A base class for `eo-grow` enum classes, implementing common properties and methods"""

    @property
    def id(self) -> int:
        """Returns an ID of an enum type

        :return: An ID
        """
        return self.values[1]

    @property
    def nice_name(self) -> str:
        """Returns nice name of an enum type

        :returns: Nice name
        """
        return self.value.capitalize()

    @property
    def color(self) -> str:
        """Returns class color

        :return: A color in hexadecimal representation
        """
        return self.values[2]

    @property
    def rgb(self) -> List[float]:
        """Returns class color in RGB representation, useful for building matplotlib colormaps

        :return: A color as [R, G, B] list of floats
        """
        return [c / 255.0 for c in self.rgb_int]

    @property
    def rgb_int(self) -> List[int]:
        """Returns class color in RGB representation

        :return: A color as [R, G, B] list of integers
        """

        hex_val = self.values[2][1:]
        return [int(hex_val[i : i + 2], 16) for i in (0, 2, 4)]

    @classmethod
    def has_value(cls, value: object) -> bool:
        """Checks if value is defined for one of the enum constants

        Example:
            LULC.has_value('forest')

        :param value: Any value
        :return: True if value is in enum and false otherwise
        """
        return value in cls._value2member_map_

    @classmethod
    def get_sentinel_hub_evaluation_function(cls, band_name: str) -> str:
        init = Template(
            "//VERSION=3 \n"
            "function setup(ds) { \n"
            "   setInputComponents([ds.$band_name]); \n"
            "   setOutputComponentCount(3); \n"
            "}\n"
            "\n"
            "function evaluatePixel(samples, scenes) {\n"
            "$script\n"
            "}\n"
        )

        script = []
        eval_pixel_script = Template("   if(samples[0].$band_name == $class_id) { return [$colors] }")
        for class_enum in cls:  # type: ignore
            colors = ", ".join([f"{c}/255" for c in class_enum.rgb_int])
            script.append(eval_pixel_script.substitute(band_name=band_name, class_id=class_enum.id, colors=colors))

        return init.substitute(band_name=band_name, script="\n".join(script))

    @classmethod
    def get_sentinel_hub_legend(cls) -> str:
        items = [{"color": class_enum.color, "label": class_enum.name} for class_enum in cls]  # type: ignore

        return json.dumps({"type": "discrete", "items": items}, indent=2)
