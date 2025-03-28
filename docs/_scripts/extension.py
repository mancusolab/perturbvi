import ast
import inspect

from griffe import Class, Docstring, dynamic_import, Extension, Function, get_logger, Object, ObjectNode
from griffe.dataclasses import Parameter


logger = get_logger(__name__)


def _get_dynamic_docstring(obj: Object, name: str) -> Docstring:
    # import object to get its evaluated docstring
    try:
        runtime_obj = dynamic_import(obj.path)
        init_docstring = getattr(runtime_obj, name).__doc__
    except ImportError:
        logger.debug(f"Could not get dynamic docstring for {obj.path}")
        return
    except AttributeError:
        logger.debug(f"Object {obj.path} does not have a __doc__ attribute")
        return

    if init_docstring is None:
        return None

    # update the object instance with the evaluated docstring
    init_docstring = inspect.cleandoc(init_docstring)

    return Docstring(init_docstring, parent=obj)


class DynamicDocstrings(Extension):
    def __init__(self, paths: list[str] | None = None) -> None:
        self.module_paths = paths

    def on_class_members(self, *, node: ast.AST | ObjectNode, cls: Class) -> None:
        logger.debug(f"Inspecting class member {cls.path}")
        if isinstance(node, ObjectNode):
            return  # skip runtime objects, their docstrings are already right
        if self.module_paths and cls.parent is None or cls.parent.path not in self.module_paths:
            return  # skip objects that were not selected

        # pull class attributes as parameters for the __init__ function...
        parameters = [
            Parameter(name=attr.name, annotation=attr.annotation, kind=attr.kind)
            for attr in cls.members.values()
            if attr.is_attribute
        ]
        # such a huge hack to pull in inherited attributes
        if cls.name != "ExpFam" and "ExpFam" in cls.parent.members:
            attr = cls.parent.members["ExpFam"].members["pred_mean"]
            parameters = [Parameter(name=attr.name, annotation=attr.annotation, kind=attr.kind)] + parameters
        cls.members["__init__"] = Function(
            name="__init__", parameters=parameters, docstring=_get_dynamic_docstring(cls, "__init__")
        )
        # add docs for __call__ only if it was explicitly defined (e.g., ExpFam, but not concrete subclasses)
        if "__call__" in cls.members:
            cls.members["__call__"].docstring = _get_dynamic_docstring(cls, "__call__")

        return
