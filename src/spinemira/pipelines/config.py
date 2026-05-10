import argparse
from functools import wraps
import inspect
import os
from pathlib import Path
import sys
from typing import Any, Callable
import yaml


def with_cli_config(
    *,
    env_map: dict[str, str] | None = None,
    config_arg: str = "config",
    log_args: bool = True,
    log_fn: Callable | None = None,
):
    """Decorator that injects args into a function

    Arguments are injected by resolving: CLI (explicit flags only) > ENV > YAML config > function defaults.

    Parameters
    ----------
    env_map : dict[str, str] | None, optional
        Map describing how arguments should be resolved from the environment, by default None
    config_arg : str, optional
        Name of the argument for resolving configuration file, by default "config"
    log_args : bool, optional
        Configures is logging of arguments should be performed, by default True
    log_fn : Callable | None, optional
        Method to call for logging, will print to standard output if unspecified, by default None
    """
    env_map = env_map or {}

    def decorator(fn: Callable):
        signature = inspect.signature(fn)
        hints = getattr(fn, "__annotations__", {}) or {}

        parser = argparse.ArgumentParser(description=fn.__doc__ or fn.__name__)
        parser.add_argument(
            f"--{config_arg.replace('_', '-')}",
            dest=config_arg,
            type=Path,
            help="YAML config file",
        )

        # Track which parameters are required by the function signature
        required_params: set[str] = {
            name
            for name, param in signature.parameters.items()
            if name != "argv"
            and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
            and param.default is inspect._empty
        }

        # Build CLI options from signature
        for name, param in signature.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if name == "argv":
                continue

            ann = hints.get(name, str)
            default = param.default
            arg_name = f"--{name.replace('_', '-')}"

            # Boolean flags (--x / --no-x)
            if _is_bool(ann) or isinstance(default, bool):
                group = parser.add_mutually_exclusive_group(required=False)
                group.add_argument(arg_name, dest=name, action="store_true")
                group.add_argument(
                    f"--no-{name.replace('_', '-')}",
                    dest=name,
                    action="store_false",
                )
                parser.set_defaults(**{name: default})
                continue
            else:
                parser.add_argument(
                    arg_name,
                    dest=name,
                    type=_arg_type(ann),
                    required=False,
                    default=default,
                )

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # If user calls fn(dataset_root=..., ...) skip parsing.
            if kwargs and "argv" not in kwargs:
                return fn(*args, **kwargs)

            argv = kwargs.pop("argv", None)
            if argv is None:
                argv = sys.argv[1:]

            if log_args:
                log_message = "Pipeline was invoked with: " + " ".join(
                    [sys.executable, *sys.argv]
                )
                if log_fn:
                    log_fn(log_message)
                else:
                    print(log_message)

            ns, _ = parser.parse_known_args(argv)
            cli = vars(ns)

            # YAML config
            yaml_cfg: dict[str, Any] = {}
            cfg_path = cli.pop(config_arg, None)
            if cfg_path:
                yaml_cfg = yaml.safe_load(Path(cfg_path).read_text()) or {}

            # Function defaults
            defaults = {
                n: p.default
                for n, p in signature.parameters.items()
                if p.default is not inspect._empty and n != "argv"
            }

            # ENV values (only if present in env)
            env_cfg: dict[str, Any] = {}
            for key, env_var in env_map.items():
                val = os.getenv(env_var)
                if val is not None:
                    env_cfg[key] = _auto_cast(val)

            # Check which CLI flags were explicitly provided
            provided = set(_provided_keys(argv))

            # Merge with precedence: defaults -> YAML -> ENV -> CLI(explicit only)
            final = dict(defaults)
            final.update(yaml_cfg)
            final.update(env_cfg)

            for k, v in cli.items():
                if k in provided:
                    final[k] = v

            # Ensure required params exist
            for name in required_params:
                if name not in final or final[name] is None:
                    raise SystemExit(f"Missing required argument: {name}")

            if log_args:
                payload = _serialize_args(final)

                log_message = "\n".join(
                    [
                        f"Passing the following arguments to {fn.__name__}:",
                        *(f"    {k}: {v}" for k, v in payload.items()),
                    ]
                )

                if log_fn:
                    log_fn(log_message)
                else:
                    print(log_message)

            return fn(**final)

        return wrapper

    return decorator


def _serialize_args(args: dict[str, Any]) -> dict[str, Any]:
    def convert(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, dict):
            return {k: convert(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [convert(x) for x in v]
        return v

    return {k: convert(v) for k, v in args.items()}


def _provided_keys(argv: list[str]) -> list[str]:
    keys: list[str] = []
    for tok in argv:
        if tok.startswith("--"):
            k = tok[2:]
            if k.startswith("no-"):
                k = k[3:]
            keys.append(k.replace("-", "_"))
    return keys


def _is_bool(t) -> bool:
    return t is bool


def _arg_type(t):
    if t in (str, int, float):
        return t
    if t is Path:
        return Path
    return str


def _auto_cast(x: str) -> Any:
    for cast in (int, float):
        try:
            return cast(x)
        except ValueError:
            pass
    if x.lower() in {"true", "false"}:
        return x.lower() == "true"
    return x
