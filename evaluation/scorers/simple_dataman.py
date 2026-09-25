"""
Simple DataMan Inference Script
A simplified version of dataman_infer.py that takes raw text as input
and outputs the DataMan overall score using RuPeng/DataMan-1.5B-EN model.
"""

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from openai import OpenAI
import re

# Request Templates
REQUEST_TEMPLATES = {
    "en": {
        "all_rating": """Please score the text on fourteen evaluation criteria and specify its domain:
Text: {text}
Domain:_
[1]Accuracy:_/5
[2]Coherence:_/5
[3]Language Consistency:_/5
[4]Semantic Density:_/5
[5]Knowledge Novelty:_/5
[6]Topic Focus:_/5
[7]Creativity:_/5
[8]Professionalism:_/5
[9]Style Consistency:_/5
[10]Grammatical Diversity:_/5
[11]Structural Standardization:_/5
[12]Originality:_/5
[13]Sensitivity:_/5
[14]Overall Score:_/5""",
        "score_only": """Please give an overall score for the text:
Text: {text}
Overall Score:_/5""",
        "detprompt_score_only": """Please give an overall score for the text:
Text: {text}
[1]Accuracy
[2]Coherence
[3]Language Consistency
[4]Semantic Density
[5]Knowledge Novelty
[6]Topic Focus
[7]Creativity
[8]Professionalism
[9]Style Consistency
[10]Grammatical Diversity
[11]Structural Standardization
[12]Originality
[13]Sensitivity
Overall Score:_/5""",
        "domain_only": """Please specify an domain type for the text:
Text: {text}
Domain:_""",
        "detprompt_domain_only": """Please specify an domain type for the text:
Text: {text}
Domain Types: [A]Medicine [B]Finance [C]Law [D]Education [E]Technology [F]Entertainment [G]Mathematics [H]Coding [I]Government [J]Culture [K]Transportation [L]Retail E-commerce [M]Telecommunication [N]Agriculture [O]Other
Domain:_""",
    },
}
# Regular expression: matches ellipses surrounded by double quotes, matches URL items starting with ![](//) characters, score, domain type, raw text, N/A
ELLIPSIS_PATTERN = r'^["“”]…+["“”]|^["“”]\.\.\.+["“”]'
LINK_PATTERN = r"!\[\]\([\/]?[^\)]+\)\n"
TEXT_PATTERNS = {
    "en": {
        "all_rating": r"\nText: (.*?)(\n)+Domain",
        "score_only": r"\nText: (.*?)(\n)+Overall Score",
        "domain_only": r"\nText: (.*?)(\n)+Domain",
    },
}


class DataManInference:
    def __init__(
        self,
        model_name="RuPeng/DataMan-1.5B-EN",
        max_tokens=29,
        temperature=0.0,
        seed=1024,
        model_type="all_rating",
        use_server=False,
        server_url="http://localhost:8000/v1",
        api_key="EMPTY",
    ):
        """
        Initialize the DataMan inference model.

        Args:
            model_name (str): The model name or path. Default is "RuPeng/DataMan-1.5B-EN"
            max_tokens (int): Maximum number of output tokens
            temperature (float): Generation temperature
            seed (int): Random seed for reproducibility
            model_type (str): Type of evaluation ('score_only', 'detprompt_score_only', 'all_rating', 'domain_only', 'detprompt_domain_only')
            use_server (bool): Whether to use an existing vLLM server instead of loading the model locally
            server_url (str): The base URL of the vLLM server (e.g., "http://localhost:8000/v1")
            api_key (str): API key for the server (use "EMPTY" for vLLM servers)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.model_type = model_type
        self.use_server = use_server
        self.server_url = server_url
        self.api_key = api_key

        # Set the request template based on model type (using English)
        self.request_template = REQUEST_TEMPLATES["en"][model_type]

        # Regular expressions for filtering
        self.ellipsis_pattern = r'^["“”]…+["“”]|^["“”]\.\.\.+["“”]'
        self.link_pattern = r"!\[\]\([\/]?[^\)]+\)\n"

        # Text truncation limits (optimized for English)
        self.truncate_max_length = 1894  # Token limit for English
        self.char_truncate_max_length = 20000  # Character limit for English

        # Initialize tokenizer and model
        self._load_model()

    def _load_model(self):
        """Load the tokenizer and LLM model or initialize OpenAI client."""
        # Always load tokenizer for text preprocessing
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left", use_fast=True
        )

        if self.use_server:
            # Initialize OpenAI client for server mode
            print(f"Initializing OpenAI client for server: {self.server_url}")
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.server_url,
            )
            print("OpenAI client initialized successfully!")
        else:
            # Initialize local vLLM model
            print(f"Loading local model: {self.model_name}")
            self.llm = LLM(model=self.model_name, tokenizer=self.model_name)

        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            seed=self.seed,
            stop_token_ids=[151643, 151645],  # Qwen2 stop tokens
            max_tokens=self.max_tokens,
        )

        print("Model loaded successfully!")

    def _preprocess_text(self, text):
        """
        Preprocess the input text by applying regular filtering and truncation.

        Args:
            text (str): Raw input text

        Returns:
            str: Preprocessed text
        """
        # Apply regular expression filtering
        text = re.sub(self.ellipsis_pattern, "", text)
        text = re.sub(self.link_pattern, "", text)

        # Truncate text if too long
        if len(text) > self.char_truncate_max_length:
            # Take first and last halves if character length exceeds limit
            mid_point = self.char_truncate_max_length // 2
            text = text[:mid_point] + text[-mid_point:]

        # Tokenize and check token length
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.truncate_max_length:
            # Truncate tokens and add ellipsis in the middle
            mid_point = self.truncate_max_length // 2
            tokens = tokens[:mid_point] + ["..."] + tokens[-mid_point:]

        # Convert tokens back to string
        text = self.tokenizer.convert_tokens_to_string(tokens).strip()

        return text

    def _create_chat_text(self, text):
        """
        Create a chat-formatted text for the model.

        Args:
            text (str): Preprocessed text

        Returns:
            str: Chat-formatted text ready for inference
        """
        request = self.request_template.format(text=text)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request},
        ]

        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return chat_text

    def _parse_output(self, output_text):
        """
        Parse the model output based on model_type.

        Args:
            output_text (str): Raw output from the model

        Returns:
            dict or str: Parsed output based on model_type
        """
        output_text = output_text.strip()

        if self.model_type.endswith("score_only"):
            # For score_only models, return just the score
            return {"overall_score": output_text}
        elif self.model_type.endswith("domain_only"):
            # For domain models, return the domain
            return {"application_domain": output_text}
        elif self.model_type == "all_rating":
            # For all_rating, parse the multi-line output
            lines = output_text.split("\n")
            result = {}

            # The expected order based on the original dataman_infer.py
            metric_names = [
                "application_domain",
                "accuracy",
                "coherence",
                "language_consistency",
                "semantic_density",
                "knowledge_novelty",
                "topic_focus",
                "creativity",
                "professionalism",
                "style_consistency",
                "grammatical_diversity",
                "structural_standardization",
                "originality",
                "sensitivity",
                "overall_score",
            ]

            # Map lines to metric names
            for i, line in enumerate(lines):
                if i < len(metric_names):
                    if i == 0:
                        result[metric_names[i]] = line.strip()
                    else:
                        try:
                            result[metric_names[i]] = float(line.strip())
                        except:
                            result[metric_names[i]] = 0.0

            return result
        else:
            # Default: return the raw output
            return output_text

    def score_text(self, text):
        """
        Score a single text and return the parsed result based on model_type.

        Args:
            text (str): Raw input text to be scored

        Returns:
            str or dict: Parsed result based on model_type
        """
        # Preprocess the text
        preprocessed_text = self._preprocess_text(text)

        # Create chat-formatted text
        chat_text = self._create_chat_text(preprocessed_text)

        # Generate response
        if self.use_server:
            # Use OpenAI client to call vLLM server
            completion = self.client.completions.create(
                model=self.model_name,
                prompt=chat_text,
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=self.max_tokens,
            )
            output_text = completion.choices[0].text.strip()
        else:
            outputs = self.llm.generate(
                [chat_text],
                self.sampling_params,
                use_tqdm=False,
            )
            output_text = outputs[0].outputs[0].text.strip()
        parsed_result = self._parse_output(output_text)

        return parsed_result

    def score_texts(self, texts):
        """
        Score multiple texts in batch.

        Args:
            texts (list): List of raw input texts to be scored

        Returns:
            list: List of parsed results based on model_type
        """
        # Preprocess all texts
        preprocessed_texts = [self._preprocess_text(text) for text in texts]

        # Create chat-formatted texts
        chat_texts = [self._create_chat_text(text) for text in preprocessed_texts]

        if self.use_server:
            completions = self.client.completions.create(
                model=self.model_name,
                prompt=chat_texts,
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=self.max_tokens,
            )
            results = [
                self._parse_output(output.text.strip())
                for output in completions.choices
            ]
        else:
            # Generate responses in batch using the local vLLM model
            outputs = self.llm.generate(
                chat_texts,
                self.sampling_params,
                use_tqdm=False,
            )
            results = [
                self._parse_output(output.outputs[0].text.strip()) 
                for output in outputs
            ]

        return results


def main():
    """Example usage of the DataManInference class with different model types."""

    # Example texts for scoring
    example_texts = [
        "This is a high-quality research paper about machine learning algorithms. It presents novel approaches to deep learning with comprehensive experimental validation and clear mathematical formulations.",
        "hello world",
        "The quick brown fox jumps over the lazy dog. This is a simple sentence used for testing purposes.",
    ]

    # Test different model types
    # model_types = ["score_only", "detprompt_score_only", "domain_only", "all_rating"]
    model_types = ["all_rating"]

    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"Testing model_type: {model_type}")
        print(f"{'='*80}")

        # Initialize the inference model with current model type
        dataman = DataManInference(model_type=model_type, use_server=True)

        # Test with first example text
        result = dataman.score_texts(example_texts)

        print(f"Input text: {example_texts[0][:100]}...")
        print(f"Result: {result}")

        print("-" * 50)


if __name__ == "__main__":
    main()
