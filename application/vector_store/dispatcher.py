from src.schemas.vector_store_schema import VectorStoreType

from .base import BaseVectorStore
from .milvus import MilvusVectorStore  # noqa: F401
from .pinecone import PineconeVectorStore
from .qdrant import QdrantVectorStore  # noqa: F401
from .chroma import ChromaVectorStore
from .exceptions import VectorStoreNotImplementedError

class VectorStoreFactory:

    @staticmethod
    def from_vector_store(
        vector_store: VectorStoreType
    ) -> BaseVectorStore:
        if vector_store == VectorStoreType.PINECONE:
            return PineconeVectorStore()
        elif vector_store == VectorStoreType.MILVUS:
            raise VectorStoreNotImplementedError("'Milvus' vector store is not implemented.")
            # return MilvusVectorStore() # TODO
        elif vector_store == VectorStoreType.QDRANT:
            raise VectorStoreNotImplementedError("'Qdrant' vector store is not implemented.")
            # return QdrantVectorStore() # TODO
        elif vector_store == VectorStoreType.CHROMA:
            raise VectorStoreNotImplementedError("'Chroma' vector store is not implemented.")
            # return ChromaVectorStore() # DONE ('completed')
        else:
            raise VectorStoreNotImplementedError(
                "unsupported vector store "
            )


class VectorStoreDispatcher:
    factory = VectorStoreFactory()

    @classmethod
    def dispatch(cls, vector_store:VectorStoreType) -> BaseVectorStore:
        """
        Dispatch the 'vector_store' instance of the given vector store type

        Args:
            vector_store (VectorStoreType)
        Returns:
            BaseVectorStore: vector store istance
        """
        vector_inst = cls.factory.from_vector_store(vector_store)
        return vector_inst
