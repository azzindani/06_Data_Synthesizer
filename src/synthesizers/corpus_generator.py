"""Corpus/document generation synthesizer."""

import os
import re
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

from .base import BaseSynthesizer
from ..core.config import Config
from ..core.logger import get_logger
from ..providers.factory import ProviderFactory
from ..validators.language import LanguageValidator


class CorpusGenerator(BaseSynthesizer):
    """Generates full documents/corpus texts like contracts, articles, etc."""

    def __init__(self, config: Config, provider: ProviderFactory = None):
        """Initialize corpus generator.

        Args:
            config: Configuration object
            provider: Optional provider factory
        """
        super().__init__(config, provider)
        self.logger = get_logger(__name__)
        self._load_domain_config()
        self.language_validator = LanguageValidator(
            self.config.quality.target_language,
            self.config.quality.language_confidence
        )

    def _load_domain_config(self):
        """Load domain-specific configuration."""
        # Default contract taxonomy
        self.taxonomy = self.config.domain.get('taxonomy', {
            'document_types': [
                'Service Agreement',
                'Non-Disclosure Agreement',
                'Employment Contract',
                'Partnership Agreement',
                'License Agreement'
            ],
            'industries': [
                'Technology',
                'Finance',
                'Healthcare',
                'Manufacturing',
                'Retail'
            ],
            'jurisdictions': [
                'US Federal Law',
                'UK Law',
                'Indonesian Law',
                'Singapore Law',
                'EU Law'
            ]
        })

        self.domain_name = self.config.domain.get('domain', 'Legal Documents')
        self.min_length = self.config.domain.get('min_document_length', 2000)

    def get_topics(self) -> List[str]:
        """Get list of document types to generate."""
        return self.taxonomy.get('document_types', ['Generic Document'])

    def get_system_prompt(self) -> str:
        """Get system prompt for corpus generation."""
        return f"""You are a senior legal professional with extensive experience drafting {self.domain_name}.

Create realistic, professional documents that:
- Use appropriate legal terminology and structure
- Include specific details (dates, amounts, parties)
- Follow proper formatting for the document type
- Are legally sound and comprehensive
- Reflect actual business practices"""

    def create_prompt(self, item: Dict) -> str:
        """Create corpus generation prompt.

        Args:
            item: Dictionary with generation parameters

        Returns:
            Prompt string
        """
        doc_type = item.get('topic', 'Service Agreement')

        # Generate random scenario
        industry = random.choice(self.taxonomy.get('industries', ['Technology']))
        jurisdiction = random.choice(self.taxonomy.get('jurisdictions', ['US Federal Law']))

        return f"""Generate a complete, realistic {doc_type} for the {industry} industry.

REQUIREMENTS:
1. Document must be at least {self.min_length} words
2. Use {jurisdiction} as governing law
3. Include all standard sections for this document type
4. Use realistic party names and details
5. Include specific terms, dates, and amounts
6. Follow proper legal document structure

DOCUMENT STRUCTURE (adapt as appropriate):
- Title and Parties
- Recitals/Background
- Definitions
- Main Terms and Conditions
- Rights and Obligations
- Warranties and Representations
- Limitation of Liability
- Termination
- Dispute Resolution
- General Provisions
- Signatures

Generate the complete document now. Be thorough and realistic.

DOCUMENT:"""

    def synthesize_item(self, item: Dict, item_id: str) -> List[Dict]:
        """Synthesize a single document.

        Args:
            item: Item data
            item_id: Unique identifier

        Returns:
            List with single document result
        """
        if self.progress.is_processed(item_id):
            return []

        prompt = self.create_prompt(item)
        system_prompt = self.get_system_prompt()

        try:
            result = self.provider.generate(prompt, system_prompt)
            self.progress.increment_requests()

            if not result.success:
                self.logger.warning(f"Generation failed: {result.finish_reason}")
                return []

            document_text = result.text.strip()

            # Validate length
            word_count = len(document_text.split())
            if word_count < self.min_length * 0.5:  # Allow some flexibility
                self.logger.warning(f"Document too short: {word_count} words")
                return []

            # Validate language
            lang_result = self.language_validator.validate_corpus(document_text)

            # Build result
            doc_result = {
                'corpus_id': item_id,
                'document_type': item.get('topic', 'Unknown'),
                'document_text': document_text,
                'word_count': word_count,
                'char_count': len(document_text),
                **lang_result,
                'timestamp': datetime.now().isoformat(),
                'provider': self.provider.current_provider_name
            }

            # Update progress
            self.progress.mark_processed(item_id)
            self.progress.add_generated(1)

            return [doc_result]

        except Exception as e:
            self.logger.error(f"Corpus generation failed: {e}")
            self.progress.add_error({
                'type': 'generation_error',
                'item_id': item_id,
                'message': str(e)
            })
            return []

    def run(self, save_interval: int = 1) -> None:
        """Run corpus generation.

        Args:
            save_interval: Save after each N documents
        """
        doc_types = self.get_topics()
        docs_per_type = self.config.synthesis.questions_per_topic

        self.logger.info(f"Starting corpus generation: {len(doc_types)} types, {docs_per_type} each")

        for type_idx, doc_type in enumerate(doc_types):
            if self._shutdown_requested:
                break

            self.logger.info(f"Generating {doc_type} ({type_idx + 1}/{len(doc_types)})")

            for i in range(docs_per_type):
                if self._shutdown_requested:
                    break

                item_id = f"{doc_type.replace(' ', '_')}_{i:04d}"

                if self.progress.is_processed(item_id):
                    continue

                item = {'topic': doc_type}
                results = self.synthesize_item(item, item_id)

                if results:
                    self._save_results(results, doc_type)
                    self.progress.save()

        self.logger.info("Corpus generation complete")

    def _save_results(self, results: List[Dict], doc_type: str) -> None:
        """Save corpus results."""
        if not results:
            return

        df = pd.DataFrame(results)

        clean_type = re.sub(r'[^\w\s-]', '', doc_type).strip().replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"corpus_{clean_type}_{timestamp}.parquet"

        output_type = self.config.output.type

        if output_type in ['local', 'both']:
            local_dir = self.config.output.local_path
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)
            df.to_parquet(local_path, index=False)
            self.logger.info(f"Saved: {local_path}")

        if output_type in ['huggingface', 'both']:
            from huggingface_hub import upload_file

            # try/finally so a failed upload doesn't leak parquet files in /tmp.
            temp_path = f"/tmp/{filename}"
            try:
                df.to_parquet(temp_path, index=False)
                upload_file(
                    path_or_fileobj=temp_path,
                    path_in_repo=filename,
                    repo_id=self.config.output.repository,
                    repo_type="dataset",
                    token=self.config.output.huggingface_token,
                    commit_message=f"Corpus: {doc_type} - {len(results)} docs"
                )
                self.logger.info(f"Uploaded: {filename}")
            except Exception as e:
                self.logger.error(f"Upload failed: {e}")
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


if __name__ == "__main__":
    print("=" * 60)
    print("CORPUS GENERATOR TEST")
    print("=" * 60)

    from ..core.config import load_config
    import os

    config = load_config("config.yaml")
    config.output.type = "local"
    config.output.local_path = "./test_output"
    config.synthesis.questions_per_topic = 2

    config.domain = {
        'domain': 'Legal Contracts',
        'min_document_length': 500,
        'taxonomy': {
            'document_types': ['NDA', 'Service Agreement'],
            'industries': ['Technology', 'Finance'],
            'jurisdictions': ['US Law', 'UK Law']
        }
    }

    try:
        gen = CorpusGenerator(config)
        print("  ✓ Corpus Generator created")

        topics = gen.get_topics()
        print(f"  ✓ Document types: {topics}")

        prompt = gen.create_prompt({'topic': 'NDA'})
        print(f"  ✓ Prompt length: {len(prompt)} chars")

        sys_prompt = gen.get_system_prompt()
        print(f"  ✓ System prompt: {len(sys_prompt)} chars")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
