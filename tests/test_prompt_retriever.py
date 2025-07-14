"""Tests for the PromptRetriever class."""

import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest
from jinja2 import TemplateNotFound

from rubric.utils.prompt_retriever import PromptRetriever, get_prompt


@pytest.fixture
def temp_prompts_dir() -> Generator[Path, None, None]:
    """Create a temporary directory with test prompt templates."""
    with tempfile.TemporaryDirectory() as temp_dir:
        prompts_dir = Path(temp_dir)

        # Create test templates
        (prompts_dir / "simple.jinja").write_text("Hello World!")
        (prompts_dir / "with_variables.jinja").write_text(
            "Hello {{name}}! You are {{age}} years old."
        )
        (prompts_dir / "multiline.jinja").write_text(
            """
This is a multiline template.
Name: {{name}}
Description: {{description}}
        """.strip()
        )
        (prompts_dir / "complex.jinja").write_text(
            """
{{context}}

Task: {{task}}
Criterion: {{criterion}}

{% if include_examples %}
Examples:
{% for example in examples %}
- {{example}}
{% endfor %}
{% endif %}

Output: {{output_format}}
        """.strip()
        )

        yield prompts_dir


@pytest.fixture
def empty_prompts_dir() -> Generator[Path, None, None]:
    """Create an empty temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def retriever_with_temp_dir(temp_prompts_dir: Path) -> PromptRetriever:
    """Create a PromptRetriever instance with temporary directory."""
    return PromptRetriever(temp_prompts_dir)


class TestPromptRetriever:
    """Test cases for PromptRetriever class."""

    def test_init_with_default_dir(self) -> None:
        """Test initialization with default prompts directory."""
        retriever = PromptRetriever()
        expected_dir = Path(__file__).parent.parent / "rubric" / "prompts"
        assert retriever.prompts_dir == expected_dir
        assert retriever.prompts_dir.exists()

    def test_init_with_custom_dir(self, temp_prompts_dir: Path) -> None:
        """Test initialization with custom prompts directory."""
        retriever = PromptRetriever(temp_prompts_dir)
        assert retriever.prompts_dir == temp_prompts_dir
        assert retriever.prompts_dir.exists()

    def test_init_with_string_path(self, temp_prompts_dir: Path) -> None:
        """Test initialization with string path."""
        retriever = PromptRetriever(str(temp_prompts_dir))
        assert retriever.prompts_dir == temp_prompts_dir

    def test_init_with_nonexistent_dir(self) -> None:
        """Test initialization with non-existent directory raises error."""
        with pytest.raises(FileNotFoundError, match="Prompts directory not found"):
            PromptRetriever("/nonexistent/path")

    def test_get_template_names(self, retriever_with_temp_dir: PromptRetriever) -> None:
        """Test getting list of template names."""
        names = retriever_with_temp_dir.get_template_names()
        expected_names = ["complex", "multiline", "simple", "with_variables"]
        assert sorted(names) == expected_names

    def test_get_template_names_empty_dir(self, empty_prompts_dir: Path) -> None:
        """Test getting template names from empty directory."""
        retriever = PromptRetriever(empty_prompts_dir)
        assert retriever.get_template_names() == []

    def test_template_exists_true(self, retriever_with_temp_dir: PromptRetriever) -> None:
        """Test template_exists returns True for existing template."""
        assert retriever_with_temp_dir.template_exists("simple")
        assert retriever_with_temp_dir.template_exists("with_variables")

    def test_template_exists_false(self, retriever_with_temp_dir: PromptRetriever) -> None:
        """Test template_exists returns False for non-existing template."""
        assert not retriever_with_temp_dir.template_exists("nonexistent")
        assert not retriever_with_temp_dir.template_exists("missing")

    def test_get_template_success(self, retriever_with_temp_dir: PromptRetriever) -> None:
        """Test successfully getting a template."""
        template = retriever_with_temp_dir.get_template("simple")
        assert template is not None
        assert template.render() == "Hello World!"

    def test_get_template_with_caching(self, retriever_with_temp_dir: PromptRetriever) -> None:
        """Test that templates are cached."""
        template1 = retriever_with_temp_dir.get_template("simple")
        template2 = retriever_with_temp_dir.get_template("simple")
        assert template1 is template2  # Same object due to caching

    def test_get_template_not_found(self, retriever_with_temp_dir: PromptRetriever) -> None:
        """Test getting non-existent template raises TemplateNotFound."""
        with pytest.raises(TemplateNotFound, match="Template 'nonexistent' not found"):
            retriever_with_temp_dir.get_template("nonexistent")

    def test_render_template_simple(self, retriever_with_temp_dir: PromptRetriever) -> None:
        """Test rendering simple template without variables."""
        result = retriever_with_temp_dir.render_template("simple")
        assert result == "Hello World!"

    def test_render_template_with_variables(self, retriever_with_temp_dir: PromptRetriever) -> None:
        """Test rendering template with variables."""
        result = retriever_with_temp_dir.render_template("with_variables", name="Alice", age=30)
        assert result == "Hello Alice! You are 30 years old."

    def test_render_template_multiline(self, retriever_with_temp_dir: PromptRetriever) -> None:
        """Test rendering multiline template."""
        result = retriever_with_temp_dir.render_template(
            "multiline", name="Bob", description="A test user"
        )
        expected = "This is a multiline template.\nName: Bob\nDescription: A test user"
        assert result == expected

    def test_render_template_complex(self, retriever_with_temp_dir: PromptRetriever) -> None:
        """Test rendering complex template with conditionals and loops."""
        result = retriever_with_temp_dir.render_template(
            "complex",
            context="Test context",
            task="Test task",
            criterion="Test criterion",
            include_examples=True,
            examples=["Example 1", "Example 2"],
            output_format="JSON",
        )

        assert "Test context" in result
        assert "Task: Test task" in result
        assert "Criterion: Test criterion" in result
        assert "Examples:" in result
        assert "- Example 1" in result
        assert "- Example 2" in result
        assert "Output: JSON" in result

    def test_render_template_complex_no_examples(
        self, retriever_with_temp_dir: PromptRetriever
    ) -> None:
        """Test rendering complex template without examples."""
        result = retriever_with_temp_dir.render_template(
            "complex",
            context="Test context",
            task="Test task",
            criterion="Test criterion",
            include_examples=False,
            output_format="JSON",
        )

        assert "Test context" in result
        assert "Examples:" not in result
        assert "Output: JSON" in result

    def test_render_template_missing_variable(
        self, retriever_with_temp_dir: PromptRetriever
    ) -> None:
        """Test rendering template with missing variable raises error."""
        with pytest.raises(Exception):  # StrictUndefined will raise an error
            retriever_with_temp_dir.render_template("with_variables", name="Alice")
            # Missing 'age' variable should raise error due to StrictUndefined

    def test_get_prompt_convenience_method(self, retriever_with_temp_dir: PromptRetriever) -> None:
        """Test get_prompt convenience method."""
        result = retriever_with_temp_dir.get_prompt("with_variables", name="Charlie", age=25)
        assert result == "Hello Charlie! You are 25 years old."

    def test_get_raw_template_content(self, retriever_with_temp_dir: PromptRetriever) -> None:
        """Test getting raw template content."""
        content = retriever_with_temp_dir.get_raw_template_content("simple")
        assert content == "Hello World!"

    def test_get_raw_template_content_with_variables(
        self, retriever_with_temp_dir: PromptRetriever
    ) -> None:
        """Test getting raw content preserves template syntax."""
        content = retriever_with_temp_dir.get_raw_template_content("with_variables")
        assert content == "Hello {{name}}! You are {{age}} years old."

    def test_get_raw_template_content_not_found(
        self, retriever_with_temp_dir: PromptRetriever
    ) -> None:
        """Test getting raw content for non-existent template."""
        with pytest.raises(FileNotFoundError, match="Template file not found"):
            retriever_with_temp_dir.get_raw_template_content("nonexistent")

    def test_list_prompts(self, retriever_with_temp_dir: PromptRetriever) -> None:
        """Test listing all prompts with their content."""
        prompts = retriever_with_temp_dir.list_prompts()

        assert len(prompts) == 4
        assert "simple" in prompts
        assert "with_variables" in prompts
        assert "multiline" in prompts
        assert "complex" in prompts

        assert prompts["simple"] == "Hello World!"
        assert prompts["with_variables"] == "Hello {{name}}! You are {{age}} years old."

    def test_list_prompts_empty_dir(self, empty_prompts_dir: Path) -> None:
        """Test listing prompts from empty directory."""
        retriever = PromptRetriever(empty_prompts_dir)
        prompts = retriever.list_prompts()
        assert prompts == {}

    def test_environment_configuration(self, retriever_with_temp_dir: PromptRetriever) -> None:
        """Test that Jinja environment is configured correctly."""
        env = retriever_with_temp_dir.env

        # Check configuration
        assert env.autoescape is False
        assert env.trim_blocks is True
        assert env.lstrip_blocks is True

        # Test that StrictUndefined is working
        template = env.from_string("{{undefined_var}}")
        with pytest.raises(Exception):
            template.render()


class TestConvenienceFunction:
    """Test cases for the convenience get_prompt function."""

    def test_get_prompt_function_with_default_retriever(self) -> None:
        """Test the module-level get_prompt convenience function."""
        # This will use the default PromptRetriever with actual prompts directory
        # We'll test with one of the actual templates
        result = get_prompt(
            "generate-scorer-function-user",
            context="Test context",
            task="Test task",
            criterion="Test criterion",
            scorer_output_format="JSON",
            generation_guidelines="Test guidelines",
        )

        assert "Test context" in result
        assert "Test task" in result
        assert "Test criterion" in result
        assert "JSON" in result
        assert "Test guidelines" in result

    @patch("rubric.utils.prompt_retriever.PromptRetriever")
    def test_get_prompt_function_creates_new_retriever(self, mock_retriever_class: Mock) -> None:
        """Test that convenience function creates new PromptRetriever instance."""
        mock_instance = mock_retriever_class.return_value
        mock_instance.get_prompt.return_value = "mocked result"

        result = get_prompt("test_template", var1="value1", var2="value2")

        # Verify PromptRetriever was instantiated
        mock_retriever_class.assert_called_once_with()

        # Verify get_prompt was called with correct arguments
        mock_instance.get_prompt.assert_called_once_with(
            "test_template", var1="value1", var2="value2"
        )

        assert result == "mocked result"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_template_with_unicode_content(self, temp_prompts_dir: Path) -> None:
        """Test handling templates with unicode content."""
        unicode_template = temp_prompts_dir / "unicode.jinja"
        unicode_content = "Hello {{name}}! 🚀 Unicode: αβγ 中文 العربية"
        unicode_template.write_text(unicode_content, encoding="utf-8")

        retriever = PromptRetriever(temp_prompts_dir)
        result = retriever.render_template("unicode", name="Test")
        assert "🚀 Unicode: αβγ 中文 العربية" in result

    def test_template_names_sorting(self, temp_prompts_dir: Path) -> None:
        """Test that template names are returned sorted."""
        # Add more templates to test sorting
        (temp_prompts_dir / "z_last.jinja").write_text("Last")
        (temp_prompts_dir / "a_first.jinja").write_text("First")
        (temp_prompts_dir / "m_middle.jinja").write_text("Middle")

        retriever = PromptRetriever(temp_prompts_dir)
        names = retriever.get_template_names()

        # Should include all templates and be sorted
        assert "a_first" in names
        assert "m_middle" in names
        assert "z_last" in names
        assert names == sorted(names)

    def test_empty_template(self, temp_prompts_dir: Path) -> None:
        """Test handling of empty template file."""
        empty_template = temp_prompts_dir / "empty.jinja"
        empty_template.write_text("")

        retriever = PromptRetriever(temp_prompts_dir)
        result = retriever.render_template("empty")
        assert result == ""

    def test_template_with_only_whitespace(self, temp_prompts_dir: Path) -> None:
        """Test template with only whitespace content."""
        whitespace_template = temp_prompts_dir / "whitespace.jinja"
        whitespace_template.write_text("   \n\t  \n   ")

        retriever = PromptRetriever(temp_prompts_dir)
        result = retriever.render_template("whitespace")
        # Due to trim_blocks and lstrip_blocks, should be empty
        assert result.strip() == ""

    def test_case_sensitive_template_names(self, temp_prompts_dir: Path) -> None:
        """Test that template names are case sensitive."""
        (temp_prompts_dir / "CamelCase.jinja").write_text("Camel case content")
        (temp_prompts_dir / "lowercase.jinja").write_text("Lower case content")

        retriever = PromptRetriever(temp_prompts_dir)

        assert retriever.template_exists("CamelCase")
        assert retriever.template_exists("lowercase")
        assert not retriever.template_exists("camelcase")  # Different case
        assert not retriever.template_exists("LOWERCASE")  # Different case

    def test_files_without_jinja_extension_ignored(self, temp_prompts_dir: Path) -> None:
        """Test that files without .jinja extension are ignored."""
        (temp_prompts_dir / "not_template.txt").write_text("Not a template")
        (temp_prompts_dir / "template.jinja").write_text("Is a template")

        retriever = PromptRetriever(temp_prompts_dir)
        names = retriever.get_template_names()

        assert "template" in names
        assert "not_template" not in names
        assert "not_template.txt" not in names
