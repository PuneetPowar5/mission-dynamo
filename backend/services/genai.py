import json

from tqdm import tqdm
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
import logging
from vertexai.generative_models import GenerativeModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiProcessor:
    def __init__(self, model_name, project):
        self.model = VertexAI(model_name=model_name, project=project)

    def generate_document_summary(self, documents: list, **args):
        if len(documents) > 10:
            chain_type = "map_reduce"
        else:
            chain_type = "stuff"

        chain = load_summarize_chain(
            chain_type=chain_type,
            llm=self.model,
            **args
        )
        return chain.run(documents)

    def count_total_tokens(self, docs: list):
        model = GenerativeModel("gemini-1.0-pro")
        total = 0
        for doc in tqdm(docs):
            total += model.count_tokens(doc.page_content).total_billable_characters
        return total
    def get_model(self):
        return self.model

# New class for retrieving YouTube Video data
class YoutubeProcessor:
    def __init__(self, genai_processor: GeminiProcessor):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )
        self.GeminiProcessor = genai_processor

    def retrieve_youtube_documents(self, video_url: str, verbose=False):
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
        docs = loader.load()
        result = self.text_splitter.split_documents(docs)

        author = result[0].metadata['author']
        length = result[0].metadata['length']
        title = result[0].metadata['title']
        total_size = len(result)
        total_billable_characters = self.GeminiProcessor.count_total_tokens(result)

        if verbose:
            logger.info(f"{author}\n{length}\n{title}\n{total_size}\n{total_billable_characters}")

        return result

    def find_key_concepts(self, documents: list, sample_size: int=0, verbose=False):
        if sample_size > len(documents):
            raise ValueError("Group size can not be larger than the number of documents")

        if sample_size == 0:
            sample_size = len(documents) // 5
            if verbose:
                logging.info("Sample Size not specified, setting number of documents per sample to 5")

        num_docs_per_groups = len(documents) // sample_size + (len(documents) % sample_size > 0)

        if num_docs_per_groups > 10:
            raise ValueError("Each group is more than 10 documents, that makes the output quality degraded."
                             "Please increase the sample size parameter")

        elif num_docs_per_groups > 5:
            logging.warn("Each group has 5 documents, this may degrade the quality of the output"
                         "increase the sample size parameter if you would like more accurate results")

        groups = [documents[i: i+num_docs_per_groups] for i in range(0, len(documents), num_docs_per_groups)]

        batch_concepts = []

        logger.info("Finding the key concepts")
        for i in tqdm(groups):
            content = ""

            for doc in i:
                content += doc.page_content

            prompt = PromptTemplate(
                template = """
                Find and Define key concepts or terms found in the text:
                {text}
                Respond in the following format as a JSON object without any backticks separating each concept with a coma:
                {{"concept": "definition", "concept": "definition", ...}}
                """,
                input_variables=["text"]
            )

            chain = prompt | self.GeminiProcessor.model

            try:
                concept = chain.invoke({"text": content})

                batch_concepts.append(concept)
                batch_cost = 0

                if verbose:
                    total_input_characters = len(content)
                    total_input_cost = (total_input_characters / 1000) * 0.000125
                    logging.info(f"Running chain on {len(groups)} documents")
                    logging.info(f"Total Input Characters: {total_input_characters}")
                    logging.info(f"Total Cost: {total_input_cost}")

                    total_output_characters = len(concept)
                    total_output_cost = (total_output_characters / 1000) * 0.000375

                    logging.info(f"Total Output Characters: {total_output_characters}")
                    logging.info(f"Total Cost: {total_output_cost}")

                    batch_cost += total_output_cost + total_input_cost
                    logging.info(f"Total Group Cost: {total_output_cost + total_input_cost}\n")

            except Exception as e:
                logging.error(f"Error processing chain: {e}")
                continue

        print(batch_concepts)

        # processed_concepts = [json.loads(concept) for concept in batch_concepts]
        processed_concepts = []
        for concept in batch_concepts:
            if concept.strip():  # Check if the string is not empty or just whitespace
                try:
                    processed_concepts.append(json.loads(concept))
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON: {concept} - Error: {e}")
            else:
                print(f"Empty string encountered in batch_concepts: {concept}")

        logging.info(f"Total Analysis Cost: ${batch_cost}")
        return processed_concepts
