from typing import Generic, Tuple, Type, TypeVar

from isort import Config
from pydantic import BaseModel


class BaseConfig(BaseModel):
    pass


# What we want:
# - Base Pipeline class
# - each Pipeline class has it's own type of config
# - each Pipeline class should be aware of the type of it's config
# + when inheriting, one can *increase* the type of the config (add fields) and this should be reflected


class TestConfig(BaseConfig):
    x: int


class TestConfig2(TestConfig):
    x: int
    y: float


class BadConfig2(BaseConfig):
    x: str
    y: float


# # ---------------------------------------------------------------------------------------------------------------------


# # class BasePipeline:

# #     def __init__(self, config: TestConfig):
# #         self.config = config


# # class TestPipeline(BasePipeline):
# #     def __init__(self, config: BadConfig2):
# #         super().__init__(config)
# #         self.config: BadConfig2 = config

# #     def execute(self) -> float:
# #         return self.config.y

# # ---------------------------------------------------------------------------------------------------------------------

# # # Cant reference own class

# # class BasePipeline():
# #     class Config(BaseConfig):
# #         x: int

# #     def __init__(self, config: self.Config):
# #         self.config = config

# # # This fixes type of config to BasePipeline.Config

# # class BasePipeline():
# #     class Config(BaseConfig):
# #         x: int

# #     def __init__(self, config: BasePipeline.Config):
# #         self.config = config


# # ---------------------------------------------------------------------------------------------------------------------

# # # Parametrize Self (doesnt work)
# # Self = TypeVar("Self")


# # class BasePipeline:
# #     class TestConfig(BaseConfig):
# #         x: int

# #     def __init__(self: Self, config: Self.TestConfig):
# #         self.config = config


# # class TestPipeline(BasePipeline[TestConfig]):
# #     def execute(self) -> int:
# #         return self.config.x


# # ---------------------------------------------------------------------------------------------------------------------

# # # # Oneshot inheritance, works but further inheritance cant change config
# # Config = TypeVar("Config", bound=BaseConfig)


# # class BasePipeline(Generic[Config]):
# #     def __init__(self, config: Config):
# #         self.config = config


# # class TestPipeline(BasePipeline[TestConfig]):
# #     def execute(self) -> int:
# #         return self.config.x


# # ---------------------------------------------------------------------------------------------------------------------

# # # WithConfig, fucks up MRO :(
# # Config = TypeVar("Config")


# # class WithConfig(Generic[Config]):
# #     def __init__(self, config: Config):
# #         self.config = config


# # class TestPipeline(WithConfig[TestConfig]):
# #     def execute(self) -> int:
# #         return self.config.x


# # class TestPipeline2(WithConfig[TestConfig2], TestPipeline):
# #     def execute(self) -> int:
# #         y = self.config.y
# #         return self.config.x


# # # fixing the MRO ruins the init


# # class TestPipeline2(TestPipeline, WithConfig[TestConfig2]):
# #     def execute(self) -> int:
# #         y = self.config.y
# #         return self.config.x

# # ---------------------------------------------------------------------------------------------------------------------

# # Manually subtpye typevar
# Config1 = TypeVar("Config1", bound=BaseConfig)
# Config2 = TypeVar("Config2", bound=TestConfig)
# Config3 = TypeVar("Config3", bound=TestConfig2)


# class BasePipeline(Generic[Config1]):
#     def __init__(self, config: Config1):
#         self.config = config


# class TestPipeline(BasePipeline[Config2]):
#     def execute(self) -> int:
#         return self.config.x


# class TestPipeline2(TestPipeline[Config3]):
#     def execute(self) -> int:
#         y = self.config.y
#         return self.config.x


# a = TestConfig(x=3)
# b = TestConfig2(x=2, y=3)

# ok = TestPipeline2(b)
# not_ok = TestPipeline2(a)

# # Test supertype
# class TestConfig3(TestConfig2):
#     z: str

# c = TestConfig3(x=1, y=2, z="a")
# maybe_ok = TestPipeline2(c)


# A = TypeVar("A")

# class Test(Generic[A]): #kind of `forall A`

#     def __init__(self, x: int):
#         pass

#     def pair(self, x: A, y: A) -> Tuple[A, A]:
#         return (x, y)

# x = Test(1).pair(1, 2)


# # Manually subtpye typevar
# Config1 = TypeVar("Config1", bound="BasePipeline.Config")
# Config2 = TypeVar("Config2", bound="TestPipeline.Config")


# class BasePipeline(Generic[Config1]):
#     class Config:
#         x: int

#     def __init__(self, config: Config1):
#         self.config = config


# class TestPipeline(BasePipeline[Config2]):
#     class Config(BasePipeline.Config):
#         y: str

#     def execute(self) -> str:
#         return self.config.y


# Class decorator?


class BasePipeline:

    config: BaseConfig

    def __init__(self, config: BaseConfig):
        self.config = config


class TestPipeline(BasePipeline):
    class NestedConfig(BaseConfig):
        y: float

    config: NestedConfig

    def execute(self) -> float:
        return self.config.y


test = TestPipeline(TestPipeline.NestedConfig(y=2))
