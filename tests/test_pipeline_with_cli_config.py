import os
from tempfile import NamedTemporaryFile
import yaml
from unittest import TestCase

from spinemira.pipelines.config import with_cli_config


class TestWithCliConfig(TestCase):
    def test_programmatic_kwargs_bypass_parsing(self):
        @with_cli_config()
        def pipeline(a: int = 1, b: str = "x"):
            return a, b

        self.assertEqual(pipeline(a=5, b="y"), (5, "y"))

    def test_yaml_supplies_required(self):
        cfg = {
            "foo": "foo foo",
            "bar": True,
        }

        with NamedTemporaryFile(mode="w") as cfg_file:
            yaml.safe_dump(cfg, cfg_file)
            cfg_file.flush()

            @with_cli_config()
            def pipeline(foo: str, bar: bool):
                return foo, bar

            foo, bar = pipeline(argv=["--config", cfg_file.name])
            self.assertEqual(foo, "foo foo")
            self.assertEqual(bar, True)

    def test_yaml_and_cli_supplies_required(self):
        cfg = {
            "bar": "bar bar",
        }

        with NamedTemporaryFile(mode="w") as cfg_file:
            yaml.safe_dump(cfg, cfg_file)
            cfg_file.flush()

            @with_cli_config()
            def pipeline(foo: str, bar: str):
                return foo, bar

            foo, bar = pipeline(argv=["--config", cfg_file.name, "--foo", "foo foo"])
            self.assertEqual(foo, "foo foo")
            self.assertEqual(bar, "bar bar")

    def test_bool_flags_cli(self):
        @with_cli_config()
        def pipeline(foo: bool = True):
            return foo

        foo = pipeline(argv=["--no-foo"])
        self.assertFalse(foo)

        foo = pipeline(argv=["--foo"])
        self.assertTrue(foo)

    def test_env(self):
        @with_cli_config(env_map={"foo": "FOO"})
        def pipeline(foo: str):
            return foo

        os.environ["FOO"] = "foo foo"

        foo = pipeline()
        self.assertEqual(foo, "foo foo")
