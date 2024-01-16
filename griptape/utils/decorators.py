import functools
import schema
from schema import Schema


CONFIG_SCHEMA = Schema({"description": str, schema.Optional("schema"): Schema})


def activity(config: dict):
    validated_config = CONFIG_SCHEMA.validate(config)

    validated_config.update({k: v for k, v in config.items() if k not in validated_config})

    if not validated_config.get("schema"):
        validated_config["schema"] = None

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        setattr(wrapper, "name", func.__name__)
        setattr(wrapper, "config", validated_config)
        setattr(wrapper, "is_activity", True)

        return wrapper

    return decorator


def serializable(cls):
    from marshmallow import class_registry
    from griptape.schemas import BaseSchema

    class_registry.register(cls.__name__, BaseSchema.from_attrs_cls(cls))

    return cls
