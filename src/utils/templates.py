"""Custom prompt templates system."""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from string import Template

from ..core.logger import get_logger


class PromptTemplate:
    """A single prompt template with variable substitution."""

    def __init__(self, template: str, name: str = "", description: str = ""):
        """Initialize template.

        Args:
            template: Template string with ${variable} placeholders
            name: Template name
            description: Template description
        """
        self.name = name
        self.description = description
        self._template = template
        self._variables = self._extract_variables(template)

    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable names from template."""
        # Match ${var} and $var patterns
        pattern = r'\$\{?(\w+)\}?'
        return list(set(re.findall(pattern, template)))

    @property
    def variables(self) -> List[str]:
        """Get list of template variables."""
        return self._variables.copy()

    def render(self, **kwargs) -> str:
        """Render template with variables.

        Args:
            **kwargs: Variable values

        Returns:
            Rendered template string
        """
        # Use safe_substitute to leave unmatched variables intact
        return Template(self._template).safe_substitute(**kwargs)

    def validate(self, **kwargs) -> List[str]:
        """Validate that all required variables are provided.

        Args:
            **kwargs: Variable values

        Returns:
            List of missing variables
        """
        return [v for v in self._variables if v not in kwargs]


class TemplateManager:
    """Manages prompt templates from files and config."""

    def __init__(self, template_dir: str = "templates"):
        """Initialize template manager.

        Args:
            template_dir: Directory containing template files
        """
        self.logger = get_logger(__name__)
        self.template_dir = Path(template_dir)
        self._templates: Dict[str, PromptTemplate] = {}
        self._system_prompts: Dict[str, str] = {}

        # Load built-in templates
        self._load_builtin_templates()

    def _load_builtin_templates(self):
        """Load built-in default templates."""
        # QA synthesis template
        self._templates['qa_synthesis'] = PromptTemplate(
            template="""Generate ${num_variants} question-answer pairs about "${topic}" in ${language}.

Requirements:
- Questions should be diverse and cover different aspects
- Answers should be comprehensive (${min_length}-${max_length} words)
- Use proper ${language} grammar and vocabulary
- Include relevant details and examples

Format your response as JSON:
[
  {"question": "...", "answer": "..."},
  ...
]""",
            name="qa_synthesis",
            description="Generate QA pairs for a topic"
        )

        # Deep thinking template
        self._templates['deep_thinking'] = PromptTemplate(
            template="""Analyze and answer the following question with detailed reasoning.

Topic: ${topic}
Question: ${question}

Provide your response in this format:
1. Initial Analysis: Break down what the question is asking
2. Research/Reasoning: Explore different perspectives and gather relevant information
3. Synthesis: Combine insights to form a comprehensive answer
4. Final Answer: Provide a clear, well-structured response in ${language}

Format as JSON:
{
  "thinking": "your detailed reasoning process",
  "answer": "your final comprehensive answer"
}""",
            name="deep_thinking",
            description="Generate answer with reasoning trace"
        )

        # Corpus generation template
        self._templates['corpus_generation'] = PromptTemplate(
            template="""Generate a ${document_type} document for the following scenario:

Industry: ${industry}
Parties: ${parties}
Purpose: ${purpose}
Special Requirements: ${requirements}

The document should:
- Be written in formal ${language}
- Follow standard ${document_type} structure
- Include all necessary clauses and provisions
- Be legally sound and comprehensive

Generate the complete document text.""",
            name="corpus_generation",
            description="Generate full document corpus"
        )

        # System prompts
        self._system_prompts['qa_expert'] = """You are an expert content creator specializing in generating high-quality educational question-answer pairs. Your responses should be accurate, informative, and well-structured."""

        self._system_prompts['legal_expert'] = """You are a legal expert with extensive knowledge of contracts, regulations, and legal procedures. Generate accurate and professionally formatted legal content."""

        self._system_prompts['researcher'] = """You are a thorough researcher who analyzes topics from multiple perspectives, considering various sources and viewpoints before forming conclusions."""

    def load_from_file(self, filepath: str) -> bool:
        """Load templates from a YAML file.

        Args:
            filepath: Path to template file

        Returns:
            True if loaded successfully
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # Load templates
            if 'templates' in data:
                for name, config in data['templates'].items():
                    template = config.get('template', '')
                    description = config.get('description', '')
                    self._templates[name] = PromptTemplate(
                        template=template,
                        name=name,
                        description=description
                    )
                    self.logger.debug(f"Loaded template: {name}")

            # Load system prompts
            if 'system_prompts' in data:
                self._system_prompts.update(data['system_prompts'])

            self.logger.info(f"Loaded templates from: {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load templates from {filepath}: {e}")
            return False

    def load_from_config(self, config: Dict[str, Any]) -> bool:
        """Load templates from config dictionary.

        Args:
            config: Configuration dictionary with templates

        Returns:
            True if loaded successfully
        """
        try:
            if 'prompt_template' in config:
                name = config.get('template_name', 'custom')
                self._templates[name] = PromptTemplate(
                    template=config['prompt_template'],
                    name=name,
                    description=config.get('template_description', '')
                )

            if 'system_prompt' in config:
                self._system_prompts['custom'] = config['system_prompt']

            return True

        except Exception as e:
            self.logger.error(f"Failed to load templates from config: {e}")
            return False

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name.

        Args:
            name: Template name

        Returns:
            PromptTemplate or None
        """
        return self._templates.get(name)

    def get_system_prompt(self, name: str) -> Optional[str]:
        """Get a system prompt by name.

        Args:
            name: System prompt name

        Returns:
            System prompt string or None
        """
        return self._system_prompts.get(name)

    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self._templates.keys())

    def list_system_prompts(self) -> List[str]:
        """List all available system prompt names."""
        return list(self._system_prompts.keys())

    def render(self, template_name: str, **kwargs) -> str:
        """Render a template by name.

        Args:
            template_name: Name of template to render
            **kwargs: Template variables

        Returns:
            Rendered template string

        Raises:
            KeyError: If template not found
        """
        template = self._templates.get(template_name)
        if not template:
            raise KeyError(f"Template not found: {template_name}")
        return template.render(**kwargs)

    def create_prompt(self, template_name: str, system_prompt_name: str = None,
                      **kwargs) -> tuple[str, str]:
        """Create complete prompt with system and user parts.

        Args:
            template_name: User prompt template name
            system_prompt_name: System prompt name
            **kwargs: Template variables

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        user_prompt = self.render(template_name, **kwargs)

        system_prompt = ""
        if system_prompt_name:
            system_prompt = self._system_prompts.get(system_prompt_name, "")

        return system_prompt, user_prompt

    def save_template(self, name: str, template: str, description: str = ""):
        """Save a new template.

        Args:
            name: Template name
            template: Template string
            description: Template description
        """
        self._templates[name] = PromptTemplate(
            template=template,
            name=name,
            description=description
        )
        self.logger.info(f"Saved template: {name}")


if __name__ == "__main__":
    print("=" * 60)
    print("TEMPLATE MANAGER TEST")
    print("=" * 60)

    # Test 1: Create manager
    manager = TemplateManager()
    print(f"  ✓ Created template manager")
    print(f"    Templates: {manager.list_templates()}")

    # Test 2: Get template
    template = manager.get_template('qa_synthesis')
    print(f"  ✓ Got template: {template.name}")
    print(f"    Variables: {template.variables}")

    # Test 3: Render template
    rendered = template.render(
        num_variants=5,
        topic="Indonesian Contract Law",
        language="Indonesian",
        min_length=100,
        max_length=500
    )
    print(f"  ✓ Rendered template ({len(rendered)} chars)")

    # Test 4: Create full prompt
    system, user = manager.create_prompt(
        'qa_synthesis',
        'qa_expert',
        num_variants=3,
        topic="Test Topic",
        language="English",
        min_length=50,
        max_length=200
    )
    print(f"  ✓ Created prompt")
    print(f"    System: {len(system)} chars")
    print(f"    User: {len(user)} chars")

    # Test 5: Save custom template
    manager.save_template(
        'custom',
        'Generate ${count} items about ${subject}',
        'Custom template'
    )
    print(f"  ✓ Saved custom template")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
