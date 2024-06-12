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

    def find_key_concepts(self, documents: list, group_size: int=2, verbose=False):
        if group_size > len(documents):
            group_size=2

        num_docs_per_groups = len(documents) // group_size + (len(documents) % group_size > 0)

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
                Respond in the following format as a string separating each concept with a coma:
                "concept": "definition"
                """,
                input_variables=["text"]
            )

            chain = prompt | self.GeminiProcessor.model

            concept = chain.invoke({"text": content})

            batch_concepts.append(concept)
            batch_cost = 0

            if verbose:
                total_input_characters = len(content)
                total_input_cost = (total_input_characters/1000) * 0.000125
                logging.info(f"Running chain on {len(groups)} documents")
                logging.info(f"Total Input Characters: {total_input_characters}")
                logging.info(f"Total Cost: {total_input_cost}")

                total_output_characters = len(concept)
                total_output_cost = (total_output_characters/1000) * 0.000375

                logging.info(f"Total Output Characters: {total_output_characters}")
                logging.info(f"Total Cost: {total_output_cost}")

                batch_cost += total_output_cost + total_input_cost
                logging.info(f"Total Group Cost: {total_output_cost + total_input_cost}\n")

        logging.info(f"Total Analysis Cost: ${batch_cost}")
        return batch_concepts
