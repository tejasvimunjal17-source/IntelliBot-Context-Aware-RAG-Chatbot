"""Project integrity tests (Day 7). All REAL: they read the actual project files.

They guard things that break a project silently on someone else's machine: dependencies
used but not declared, environment variables that are documented but unused (or used but
undocumented), settings that Streamlit Secrets would not forward, secrets in the repository,
missing documentation, and the sample-document requirement from the official brief.
"""

import ast
import re
import sys
import unittest
from pathlib import Path

from core.config import DEFAULT_MAX_UPLOAD_MB, PROJECT_ROOT, SAMPLE_DOCS_DIR
from core.document_loader import load_path

ROOT = PROJECT_ROOT
STDLIB = set(sys.stdlib_module_names)
LOCAL_MODULES = {"core", "tests", "app", "scripts"}
# import name -> distribution name
ALIASES = {"dotenv": "python-dotenv", "fitz": "pymupdf", "sentence_transformers": "sentence-transformers"}
OPTIONAL = {"pysqlite3": "pysqlite3-binary"}  # imported only if present (see README, Troubleshooting)


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def requirement_names(filename):
    names = set()
    for line in read(filename).splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            names.add(re.split(r"[<>=!~ ;\[]", line)[0].lower().replace("_", "-"))
    return names


def imported_modules(path):
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found - STDLIB - LOCAL_MODULES


class DependencyTests(unittest.TestCase):
    def test_every_requirements_file_is_valid(self):
        try:
            from packaging.requirements import Requirement
        except ImportError:
            self.skipTest("the 'packaging' library is not installed")
        for filename in ("requirements.txt", "requirements-dev.txt", "requirements-eval.txt"):
            for line in read(filename).splitlines():
                line = line.split("#")[0].strip()
                if line and not line.startswith("-"):
                    with self.subTest(file=filename, line=line):
                        Requirement(line)  # raises InvalidRequirement if malformed

    def test_every_package_the_app_imports_is_declared_in_requirements(self):
        runtime = requirement_names("requirements.txt")
        evaluation = requirement_names("requirements-eval.txt")
        for path in [ROOT / "app.py", *sorted((ROOT / "core").glob("*.py"))]:
            for module in imported_modules(path):
                if module in OPTIONAL:
                    continue
                dist = ALIASES.get(module, module).replace("_", "-")
                with self.subTest(file=path.name, module=module):
                    self.assertTrue(dist in runtime or (module == "ragas" and dist in evaluation),
                                    f"{path.name} imports '{module}' but '{dist}' is not declared")

    def test_ragas_is_only_a_lazy_optional_import(self):
        self.assertNotIn("ragas", requirement_names("requirements.txt"))  # heavy and optional
        self.assertIn("ragas", requirement_names("requirements-eval.txt"))
        tree = ast.parse(read("core/ragas_eval.py"))
        top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any("ragas" in ast.dump(n) and "core" not in ast.dump(n) and "importlib" not in ast.dump(n)
                             for n in top_level), "ragas must be imported inside run_ragas(), not at module level")

    def test_test_and_script_dependencies_are_declared_for_developers(self):
        dev = requirement_names("requirements-dev.txt")
        self.assertIn("pytest", dev)
        self.assertIn("reportlab", dev)  # scripts/generate_sample_docs.py

    def test_the_brief_mandated_libraries_are_declared(self):
        names = requirement_names("requirements.txt")
        for required in ("streamlit", "python-dotenv", "pymupdf", "chromadb", "sentence-transformers"):
            self.assertIn(required, names)


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config_vars = set(re.findall(r'environ\.get\("([A-Z_]+)"\)', read("core/config.py")))
        self.example_vars = {m.group(1) for m in re.finditer(r"^([A-Z_]+)=", read(".env.example"), re.M)}
        secret_block = read("app.py").split("SECRET_NAMES = (")[1].split(")")[0]
        self.secret_names = set(re.findall(r'"([A-Z_]{4,})"', secret_block))

    def test_every_setting_the_code_reads_is_documented_in_env_example(self):
        self.assertEqual(sorted(self.config_vars - self.example_vars), [])

    def test_env_example_has_no_dead_entries(self):
        self.assertEqual(sorted(self.example_vars - self.config_vars), [])

    def test_every_setting_can_be_provided_through_streamlit_secrets(self):
        self.assertEqual(sorted(self.config_vars - self.secret_names), [])

    def test_every_setting_is_explained_in_the_readme(self):
        readme = read("README.md")
        self.assertEqual(sorted(v for v in self.config_vars if v not in readme), [])

    def test_streamlits_own_upload_cap_matches_the_default_upload_limit(self):
        match = re.search(r"maxUploadSize\s*=\s*(\d+)", read(".streamlit/config.toml"))
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), DEFAULT_MAX_UPLOAD_MB)

    def test_the_upload_cap_trap_is_documented(self):
        self.assertIn("maxUploadSize", read(".env.example"))
        self.assertIn("maxUploadSize", read("README.md"))


class SecretsTests(unittest.TestCase):
    PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{10,}|hf_[A-Za-z0-9]{10,}|AKIA[0-9A-Z]{12,}|ghp_[A-Za-z0-9]{20,}")

    def test_env_example_contains_only_placeholders(self):
        for line in read(".env.example").splitlines():
            if re.match(r"^[A-Z_]*(KEY|TOKEN)=", line):
                with self.subTest(line=line):
                    self.assertRegex(line.split("=", 1)[1], r"^your-")
        self.assertIsNone(self.PATTERN.search(read(".env.example")))

    def test_gitignore_protects_secrets_and_generated_data(self):
        ignored = read(".gitignore")
        for entry in (".env", "chroma_db/", ".venv/", "__pycache__/", "*.zip"):
            self.assertIn(entry, ignored)

    def test_no_real_looking_keys_anywhere_outside_the_tests(self):
        for path in ROOT.rglob("*"):
            if (path.is_file() and path.suffix in {".py", ".md", ".txt", ".toml", ".example", ".json", ""}
                    and "tests" not in path.relative_to(ROOT).parts and "__pycache__" not in path.parts
                    and ".git" not in path.parts and path.name != ".env"):
                with self.subTest(file=str(path.relative_to(ROOT))):
                    self.assertIsNone(self.PATTERN.search(path.read_text(encoding="utf-8", errors="ignore")))

    def test_the_fake_keys_used_in_tests_are_obviously_fake(self):
        for path in (ROOT / "tests").glob("*.py"):
            for match in self.PATTERN.findall(path.read_text(encoding="utf-8")):
                with self.subTest(file=path.name, key=match):
                    self.assertRegex(match, r"(?i)fake|test|other|secret")


class DocumentationTests(unittest.TestCase):
    REQUIRED = ["README.md", "docs/PROGRESS.md", "docs/GITHUB_UPLOAD_GUIDE.md", "docs/DEMO_GUIDE.md",
                "docs/EVALUATION_REPORT.md", ".env.example", "requirements.txt", "requirements-dev.txt",
                "requirements-eval.txt", ".gitignore", "app.py", "data/eval/eval_cases.json"]

    def test_all_required_files_exist(self):
        for name in self.REQUIRED:
            with self.subTest(file=name):
                self.assertTrue((ROOT / name).is_file())

    def test_the_progress_log_keeps_the_history_of_every_day(self):
        progress = read("docs/PROGRESS.md")
        for day in range(1, 8):
            with self.subTest(day=day):
                self.assertRegex(progress, rf"(?m)^## Day {day} ")

    def test_local_links_in_the_readme_point_to_real_files(self):
        for target in re.findall(r"\]\(((?:docs/|data/|scripts/)[^)#]+)\)", read("README.md")):
            with self.subTest(link=target):
                self.assertTrue((ROOT / target).exists())

    def test_the_commands_in_the_readme_match_real_files(self):
        readme = read("README.md")
        for command, target in (("streamlit run app.py", "app.py"),
                                ("python scripts/run_evaluation.py", "scripts/run_evaluation.py"),
                                ("python scripts/generate_sample_docs.py", "scripts/generate_sample_docs.py")):
            with self.subTest(command=command):
                self.assertIn(command, readme)
                self.assertTrue((ROOT / target).is_file())

    def test_the_readme_states_the_python_version_and_the_supported_formats(self):
        readme = read("README.md")
        self.assertIn("3.11", readme)
        self.assertIn("PDF", readme)
        self.assertIn("TXT", readme)

    def test_the_readme_has_the_architecture_diagram_required_by_the_brief(self):
        self.assertIn("```mermaid", read("README.md"))

    def test_the_progress_log_and_readme_state_what_was_not_tested_for_real(self):
        for name in ("README.md", "docs/PROGRESS.md"):
            with self.subTest(file=name):
                text = read(name)
                self.assertIn("real ChromaDB", text)
                self.assertRegex(text, r"(?i)not (yet )?(been )?(run|tested|verified|seen)")


class BriefRequirementsTests(unittest.TestCase):
    """The official brief asks for 3-5 sample knowledge-base PDFs and a graceful PDF failure."""

    def test_three_to_five_sample_pdfs_exist(self):
        pdfs = [p for p in SAMPLE_DOCS_DIR.iterdir() if p.suffix.lower() == ".pdf"]
        self.assertTrue(3 <= len(pdfs) <= 5, len(pdfs))

    def test_every_sample_document_loads(self):
        for path in sorted(SAMPLE_DOCS_DIR.iterdir()):
            with self.subTest(file=path.name):
                self.assertTrue(load_path(path).ok)

    def test_the_sample_documents_are_labelled_as_fictional(self):
        for path in sorted(SAMPLE_DOCS_DIR.glob("*.pdf")):
            self.assertIn("fictional", load_path(path).document.text.lower())


if __name__ == "__main__":
    unittest.main()
