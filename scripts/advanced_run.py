#!/usr/bin/env python3
"""Advanced synthesis runner matching original notebook workflow."""

import os
import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import load_config, _create_default_config
from src.core.logger import get_logger, setup_root_logger
from src.core.progress import ProgressManager
from src.providers.factory import create_provider
from src.validators.language import LanguageValidator
from src.utils.json_parser import extract_json, parse_json_safely


class AdvancedSynthesizer:
    """Advanced QA synthesizer matching original notebook functionality."""

    def __init__(self, config=None):
        """Initialize synthesizer.

        Args:
            config: Configuration object
        """
        self.logger = get_logger(__name__)
        self.config = config or self._load_or_create_config()

        # Domain configuration (matching notebook style)
        self.domain_config = {
            'domain': 'Indonesian Legal System',
            'target_language': 'id',
            'num_variants': 5,
            'topics': []
        }

        # Initialize components
        self.progress_manager = None
        self.provider = None
        self.language_validator = None

        # Load or prompt for configuration
        self._initialize()

    def _load_or_create_config(self):
        """Load config or create default."""
        config_path = "config.yaml"
        if os.path.exists(config_path):
            return load_config(config_path)
        return _create_default_config()

    def _initialize(self):
        """Initialize all components."""
        # Ensure HuggingFace repository exists
        self._ensure_repository()

        # Initialize progress manager
        self.progress_manager = ProgressManager(self.config)

        # Load topics from progress or prompt user
        self._load_or_prompt_topics()

        # Initialize provider
        try:
            self.provider = create_provider(self.config)
            self.logger.info(f"Provider initialized: {self.provider.current_provider_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize provider: {e}")
            raise

        # Initialize language validator
        self.language_validator = LanguageValidator(
            target_language=self.domain_config['target_language'],
            min_confidence=self.config.quality.language_confidence
        )

    def _ensure_repository(self):
        """Ensure HuggingFace repository exists, create if needed."""
        if not self.config.output.repository or not self.config.output.huggingface_token:
            self.logger.warning("HuggingFace not configured, using local storage only")
            return

        try:
            from huggingface_hub import HfApi, create_repo

            api = HfApi(token=self.config.output.huggingface_token)
            repo_id = self.config.output.repository

            # Try to get repo info
            try:
                api.repo_info(repo_id=repo_id, repo_type="dataset")
                self.logger.info(f"Repository exists: {repo_id}")
            except Exception:
                # Create repository
                self.logger.info(f"Creating repository: {repo_id}")
                create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=False,
                    token=self.config.output.huggingface_token
                )
                self.logger.info(f"Repository created: {repo_id}")

        except Exception as e:
            self.logger.warning(f"Could not verify/create repository: {e}")

    def _load_or_prompt_topics(self):
        """Load topics from progress or prompt user for input."""
        # Check if topics exist in progress
        topics_file = Path(self.config.output.local_path) / "topics.json"

        # Try to load from local
        if topics_file.exists():
            try:
                with open(topics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.domain_config.update(data)
                self.logger.info(f"Loaded {len(self.domain_config['topics'])} topics from local")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load local topics: {e}")

        # Try to load from HuggingFace
        if self.config.output.repository and self.config.output.huggingface_token:
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(
                    repo_id=self.config.output.repository,
                    filename="topics.json",
                    repo_type="dataset",
                    token=self.config.output.huggingface_token
                )
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.domain_config.update(data)
                self.logger.info(f"Loaded {len(self.domain_config['topics'])} topics from HuggingFace")
                return
            except Exception:
                pass

        # Prompt user for topics
        self._prompt_for_topics()

    def _prompt_for_topics(self):
        """Interactive prompt for topic configuration."""
        print("\n" + "=" * 60)
        print("DOMAIN CONFIGURATION")
        print("=" * 60)

        # Domain name
        default_domain = self.domain_config['domain']
        domain = input(f"\nDomain name [{default_domain}]: ").strip()
        if domain:
            self.domain_config['domain'] = domain

        # Target language
        default_lang = self.domain_config['target_language']
        lang = input(f"Target language code [{default_lang}]: ").strip()
        if lang:
            self.domain_config['target_language'] = lang

        # Synthesis type
        print("\nSynthesis types:")
        print("  1. QA Pairs (question-answer)")
        print("  2. Deep Thinking (question-reasoning-answer)")
        print("  3. Corpus Documents (full documents)")

        synth_type = input("Select type [1]: ").strip() or "1"
        self.domain_config['synthesis_type'] = {
            '1': 'qa', '2': 'deep_thinking', '3': 'corpus'
        }.get(synth_type, 'qa')

        # Number of variants
        default_variants = self.domain_config['num_variants']
        variants = input(f"Items per API call [{default_variants}]: ").strip()
        if variants.isdigit():
            self.domain_config['num_variants'] = int(variants)

        # Topics
        print("\nEnter topics (one per line, empty line to finish):")
        print("Example topics for Indonesian Legal:")
        print("  - Hukum Pidana dan Sanksi")
        print("  - Hukum Perdata dan Kontrak")
        print("  - Hukum Bisnis dan Perdagangan")
        print("")

        topics = []
        while True:
            topic = input("Topic: ").strip()
            if not topic:
                break
            topics.append(topic)

        if topics:
            self.domain_config['topics'] = topics
        else:
            # Default Indonesian Legal topics
            self.domain_config['topics'] = [
                'Hukum Pidana dan Sanksi',
                'Hukum Perdata dan Kontrak',
                'Hukum Tata Negara dan Konstitusi',
                'Hukum Keluarga dan Perkawinan',
                'Hukum Bisnis dan Perdagangan',
                'Hukum Ketenagakerjaan',
                'Hukum Lingkungan',
                'Hukum HAM dan Kebebasan Sipil',
                'Prosedur Pengadilan',
                'Hukum Properti dan Tanah'
            ]
            print(f"\nUsing default topics ({len(self.domain_config['topics'])} topics)")

        # Questions per topic
        default_per_topic = self.config.synthesis.questions_per_topic
        per_topic = input(f"Questions per topic [{default_per_topic}]: ").strip()
        if per_topic.isdigit():
            self.config.synthesis.questions_per_topic = int(per_topic)

        # Save topics
        self._save_topics()

        print(f"\nConfiguration saved!")
        print(f"  Domain: {self.domain_config['domain']}")
        print(f"  Language: {self.domain_config['target_language']}")
        print(f"  Topics: {len(self.domain_config['topics'])}")
        print(f"  Type: {self.domain_config['synthesis_type']}")

    def _save_topics(self):
        """Save topics configuration to local and HuggingFace."""
        # Save locally
        local_dir = Path(self.config.output.local_path)
        local_dir.mkdir(parents=True, exist_ok=True)
        topics_file = local_dir / "topics.json"

        with open(topics_file, 'w', encoding='utf-8') as f:
            json.dump(self.domain_config, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Topics saved locally: {topics_file}")

        # Upload to HuggingFace
        if self.config.output.repository and self.config.output.huggingface_token:
            try:
                from huggingface_hub import upload_file
                upload_file(
                    path_or_fileobj=str(topics_file),
                    path_in_repo="topics.json",
                    repo_id=self.config.output.repository,
                    repo_type="dataset",
                    token=self.config.output.huggingface_token,
                    commit_message="Update topics configuration"
                )
                self.logger.info("Topics saved to HuggingFace")
            except Exception as e:
                self.logger.warning(f"Failed to upload topics: {e}")

    def create_synthesis_prompt(self, topic: str) -> str:
        """Create synthesis prompt matching notebook style.

        Args:
            topic: Topic to generate for

        Returns:
            Formatted prompt string
        """
        num_variants = self.domain_config['num_variants']
        language = self.domain_config['target_language']

        # Build JSON template
        json_format = []
        for i in range(num_variants):
            json_format.append(f'  {{"question": "pertanyaan {i+1} tentang {topic}", "answer": "jawaban lengkap dan komprehensif {i+1}"}}')

        json_template = "[\n" + ",\n".join(json_format) + "\n]"

        # Language-specific prompt
        if language == 'id':
            return f"""Buat {num_variants} pasangan pertanyaan-jawaban berkualitas tinggi tentang topik: "{topic}"

PERSYARATAN:
1. Setiap jawaban minimal 100 kata, maksimal 1500 kata
2. Gunakan bahasa Indonesia yang formal dan tepat
3. Sertakan rujukan hukum yang relevan (pasal, undang-undang, peraturan)
4. Gunakan terminologi hukum Indonesia yang benar
5. Berikan penjelasan yang komprehensif dan akurat
6. Setiap QA pair berbeda pendekatan tapi relevan dengan topik
7. Variasi tingkat kesulitan (dasar hingga lanjutan)

JENIS VARIASI:
- Definisi dan konsep dasar
- Prosedur dan mekanisme hukum
- Analisis kasus praktis
- Perbandingan dengan peraturan lain
- Aplikasi dalam kehidupan sehari-hari

FORMAT JSON:
{json_template}

Berikan HANYA JSON array yang valid tanpa penjelasan tambahan."""
        else:
            # Generic English prompt
            return f"""Create {num_variants} high-quality question-answer pairs about: "{topic}"

REQUIREMENTS:
1. Each answer should be 100-1500 words
2. Use formal, precise language
3. Include relevant references and citations
4. Provide comprehensive and accurate explanations
5. Each QA pair should have different approach but relevant to topic
6. Vary difficulty levels (basic to advanced)

VARIATION TYPES:
- Definitions and basic concepts
- Procedures and mechanisms
- Practical case analysis
- Comparisons with other regulations
- Real-world applications

JSON FORMAT:
{json_template}

Provide ONLY valid JSON array without additional explanation."""

    def get_system_prompt(self) -> str:
        """Get system prompt based on domain.

        Returns:
            System prompt string
        """
        domain = self.domain_config['domain']
        language = self.domain_config['target_language']

        if language == 'id' and 'Legal' in domain:
            return '''Anda adalah seorang ahli hukum Indonesia yang berpengalaman dengan pemahaman mendalam tentang sistem hukum Indonesia. Tugas Anda adalah membuat pasangan pertanyaan-jawaban berkualitas tinggi untuk membantu orang memahami berbagai aspek hukum Indonesia.'''
        else:
            return f'''You are an expert in {domain} with deep knowledge and experience. Your task is to create high-quality question-answer pairs to help people understand various aspects of this domain.'''

    def synthesize_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Generate QA pairs for a topic.

        Args:
            topic: Topic to synthesize

        Returns:
            List of generated QA pairs
        """
        self.logger.info(f"Synthesizing: {topic}")

        # Check existing progress
        topic_progress = self.progress_manager.progress_data['processed_topics'].get(topic, {})
        if isinstance(topic_progress, dict):
            already_done = topic_progress.get('completed', 0)
        else:
            already_done = int(topic_progress) if topic_progress else 0
        target_count = self.config.synthesis.questions_per_topic

        if already_done >= target_count:
            self.logger.info(f"Topic already complete: {already_done}/{target_count}")
            return []

        results = []
        questions_needed = target_count - already_done
        max_attempts = 15
        attempts = 0

        self.logger.info(f"Need {questions_needed} more QA pairs")

        while len(results) < questions_needed and attempts < max_attempts:
            # Generate batch
            prompt = self.create_synthesis_prompt(topic)
            system_prompt = self.get_system_prompt()

            try:
                # Rate limiting
                time.sleep(self.config.retry.base_delay)

                result = self.provider.generate(prompt, system_prompt)
                attempts += 1

                if not result.success:
                    self.logger.warning(f"Generation failed: {result.finish_reason}")
                    continue

                # Parse JSON response
                json_content = extract_json(result.text)
                if not json_content:
                    self.logger.warning("No JSON found in response")
                    continue

                variants = parse_json_safely(json_content)
                if not isinstance(variants, list):
                    self.logger.warning("Response not a list")
                    continue

                self.logger.info(f"Generated {len(variants)} variants")

                # Process variants
                for variant in variants:
                    if len(results) >= questions_needed:
                        break

                    question = variant.get('question', '').strip()
                    answer = variant.get('answer', '').strip()

                    # Basic validation
                    if len(question) < 10 or len(answer) < 50:
                        continue

                    # Language validation
                    lang_result = self.language_validator.validate_qa_pair(question, answer)

                    # Accept if passes or has reasonable content
                    accept = (
                        lang_result.get('overall_valid', False) or
                        (len(question.split()) > 5 and len(answer.split()) > 20)
                    )

                    if not accept:
                        continue

                    # Build result
                    qa_result = {
                        'topic': topic,
                        'question': question,
                        'answer': answer,
                        'question_length': len(question),
                        'answer_length': len(answer),
                        'question_word_count': len(question.split()),
                        'answer_word_count': len(answer.split()),
                        'timestamp': datetime.now().isoformat(),
                        **lang_result
                    }

                    results.append(qa_result)

                # Update progress
                current_total = already_done + len(results)
                self.progress_manager.mark_topic_progress(
                    topic, current_total, self.config.synthesis.questions_per_topic
                )
                self.progress_manager.add_generated(len(results))
                self.progress_manager.save()

                self.logger.info(f"Progress: {len(results)}/{questions_needed}")

            except Exception as e:
                self.logger.error(f"Error: {e}")
                attempts += 1

        self.logger.info(f"Completed {topic}: {len(results)} QA pairs")
        return results

    def save_topic_chunk(self, topic: str, results: List[Dict[str, Any]]):
        """Save results for a topic.

        Args:
            topic: Topic name
            results: List of QA results
        """
        if not results:
            return

        try:
            import pandas as pd

            df = pd.DataFrame(results)

            # Clean filename
            clean_topic = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"topic_{clean_topic}_{timestamp}.parquet"

            # Save locally
            local_dir = Path(self.config.output.local_path)
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / filename
            df.to_parquet(local_path, index=False)

            self.logger.info(f"Saved locally: {filename}")

            # Upload to HuggingFace
            if self.config.output.repository and self.config.output.huggingface_token:
                from huggingface_hub import upload_file
                upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=filename,
                    repo_id=self.config.output.repository,
                    repo_type="dataset",
                    token=self.config.output.huggingface_token,
                    commit_message=f"Topic: {topic} - {len(results)} QA pairs"
                )
                self.logger.info(f"Uploaded to HuggingFace: {filename}")

        except Exception as e:
            self.logger.error(f"Failed to save chunk: {e}")

    def continue_synthesis(self):
        """Continue synthesis from where we left off."""
        self.show_progress()

        # Find incomplete topics
        incomplete = []
        for topic in self.domain_config['topics']:
            topic_progress = self.progress_manager.progress_data['processed_topics'].get(topic, {})
            if isinstance(topic_progress, dict):
                done = topic_progress.get('completed', 0)
            else:
                done = int(topic_progress) if topic_progress else 0
            if done < self.config.synthesis.questions_per_topic:
                incomplete.append(topic)

        if not incomplete:
            print("\n✓ All topics completed!")
            return

        print(f"\nContinuing synthesis for {len(incomplete)} topics...")

        for topic in incomplete:
            try:
                results = self.synthesize_topic(topic)
                if results:
                    self.save_topic_chunk(topic, results)
            except Exception as e:
                self.logger.error(f"Error on {topic}: {e}")
                continue

    def show_progress(self):
        """Display current progress."""
        total_expected = len(self.domain_config['topics']) * self.config.synthesis.questions_per_topic
        completed = self.progress_manager.progress_data.get('total_generated', 0)
        percentage = (completed / total_expected * 100) if total_expected > 0 else 0

        print("\n" + "=" * 60)
        print("SYNTHESIS PROGRESS")
        print("=" * 60)
        print(f"Overall: {completed:,}/{total_expected:,} ({percentage:.1f}%)")

        completed_topics = 0
        for t in self.domain_config['topics']:
            tp = self.progress_manager.progress_data['processed_topics'].get(t, {})
            done = tp.get('completed', 0) if isinstance(tp, dict) else (int(tp) if tp else 0)
            if done >= self.config.synthesis.questions_per_topic:
                completed_topics += 1
        print(f"Topics: {completed_topics}/{len(self.domain_config['topics'])}")
        print("")

        for topic in self.domain_config['topics']:
            tp = self.progress_manager.progress_data['processed_topics'].get(topic, {})
            done = tp.get('completed', 0) if isinstance(tp, dict) else (int(tp) if tp else 0)
            total = self.config.synthesis.questions_per_topic
            pct = (done / total * 100) if total > 0 else 0
            status = "✓" if done >= total else "→"
            print(f"  {status} {topic}: {done}/{total} ({pct:.1f}%)")

    def show_config(self):
        """Display current configuration."""
        print("\n" + "=" * 60)
        print("CURRENT CONFIGURATION")
        print("=" * 60)
        print(f"Domain: {self.domain_config['domain']}")
        print(f"Language: {self.domain_config['target_language']}")
        print(f"Questions per topic: {self.config.synthesis.questions_per_topic}")
        print(f"Variants per request: {self.domain_config['num_variants']}")
        print(f"Total expected: {len(self.domain_config['topics']) * self.config.synthesis.questions_per_topic:,}")
        print(f"\nTopics ({len(self.domain_config['topics'])}):")
        for i, topic in enumerate(self.domain_config['topics'], 1):
            print(f"  {i}. {topic}")


def main():
    """Main entry point."""
    setup_root_logger('INFO')

    print("=" * 60)
    print("ADVANCED DATA SYNTHESIZER")
    print("=" * 60)

    # Initialize synthesizer
    synthesizer = AdvancedSynthesizer()

    # Show config and start
    synthesizer.show_config()

    print("\n" + "=" * 60)
    print("Starting synthesis...")
    print("=" * 60)

    try:
        synthesizer.continue_synthesis()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        synthesizer.progress_manager.save()
    except Exception as e:
        print(f"\nError: {e}")
        synthesizer.progress_manager.save()
        raise

    print("\n" + "=" * 60)
    print("SYNTHESIS COMPLETE")
    print("=" * 60)
    synthesizer.show_progress()


if __name__ == "__main__":
    main()
