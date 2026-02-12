"""FastAPI application for Embedding Service"""

import base64
import io
import uuid
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config.settings import EmbeddingServiceConfig
from ..core.embedding_manager import EmbeddingManager
from ..core.errors import EmbeddingNotFoundError, EmbeddingServiceError, InvalidCategoryError


class AddObjectRequest(BaseModel):
    """Request model for adding object"""

    category: str = Field(..., description="Category (face, brand, car, place, etc.)")
    name: Optional[str] = Field(None, description="Object name/label")
    embedding_model: Optional[str] = Field(None, description="Override embedding model")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    image_url: Optional[str] = Field(None, description="URL to original image")


class SearchRequest(BaseModel):
    """Request model for search"""

    category: str = Field(..., description="Category to search in")
    embedding: Optional[List[float]] = Field(None, description="Query embedding vector")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    similarity_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity")


class UpdateRequest(BaseModel):
    """Request model for updating object"""

    name: Optional[str] = Field(None, description="New name")
    metadata: Optional[Dict[str, Any]] = Field(None, description="New metadata")
    image_url: Optional[str] = Field(None, description="New image URL")


class BatchAddRequest(BaseModel):
    """Request model for batch add"""

    category: str = Field(..., description="Category")
    names: Optional[List[Optional[str]]] = Field(None, description="Names for each object")
    metadata_list: Optional[List[Optional[Dict[str, Any]]]] = Field(None, description="Metadata for each object")
    image_urls: Optional[List[Optional[str]]] = Field(None, description="URLs for each object")


def create_app(config: Optional[EmbeddingServiceConfig] = None) -> FastAPI:
    """Create FastAPI application"""
    if config is None:
        config = EmbeddingServiceConfig()

    app = FastAPI(
        title="Embedding Service",
        description="Unified Embedding Service API",
        version="1.0.0",
    )

    # Initialize embedding manager
    manager = EmbeddingManager(config)

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        manager.close()

    def _decode_image(image_data: bytes) -> np.ndarray:
        """Decode image from bytes"""
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        return img

    @app.post("/objects/add")
    async def add_object(
        category: str = Form(...),
        name: Optional[str] = Form(None),
        metadata: Optional[str] = Form(None),  # JSON string
        image_url: Optional[str] = Form(None),
        image: UploadFile = File(...),
    ):
        """Add object to embedding service"""
        try:
            # Read image
            image_data = await image.read()
            img = _decode_image(image_data)

            # Parse metadata
            metadata_dict = None
            if metadata:
                import json

                metadata_dict = json.loads(metadata)

            # Add object
            object_id = manager.add(
                category=category,
                image=img,
                name=name,
                metadata=metadata_dict,
                image_url=image_url,
            )

            return {"id": str(object_id), "status": "success"}

        except InvalidCategoryError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except EmbeddingServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    @app.post("/objects/batch_add")
    async def batch_add(
        category: str = Form(...),
        names: Optional[str] = Form(None),  # JSON array
        metadata_list: Optional[str] = Form(None),  # JSON array
        image_urls: Optional[str] = Form(None),  # JSON array
        images: List[UploadFile] = File(...),
    ):
        """Batch add objects"""
        try:
            import json

            # Decode images
            decoded_images = []
            for img_file in images:
                image_data = await img_file.read()
                img = _decode_image(image_data)
                decoded_images.append(img)

            # Parse optional lists
            names_list = None
            if names:
                names_list = json.loads(names)

            metadata_list_dict = None
            if metadata_list:
                metadata_list_dict = json.loads(metadata_list)

            urls_list = None
            if image_urls:
                urls_list = json.loads(image_urls)

            # Batch add
            ids = manager.batch_add(
                category=category,
                images=decoded_images,
                names=names_list,
                metadata_list=metadata_list_dict,
                image_urls=urls_list,
            )

            return {"ids": [str(id) for id in ids], "status": "success"}

        except InvalidCategoryError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except EmbeddingServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    @app.get("/objects/{object_id}")
    async def get_object(object_id: str):
        """Get object by ID"""
        try:
            obj_uuid = uuid.UUID(object_id)
            obj = manager.get(obj_uuid)

            if obj is None:
                raise HTTPException(status_code=404, detail="Object not found")

            return obj

        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid UUID format")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    @app.delete("/objects/{object_id}")
    async def delete_object(object_id: str):
        """Delete object"""
        try:
            obj_uuid = uuid.UUID(object_id)
            deleted = manager.delete(obj_uuid)

            if not deleted:
                raise HTTPException(status_code=404, detail="Object not found")

            return {"status": "success", "id": object_id}

        except EmbeddingNotFoundError:
            raise HTTPException(status_code=404, detail="Object not found")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid UUID format")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    @app.patch("/objects/{object_id}")
    async def update_object(
        object_id: str,
        name: Optional[str] = Form(None),
        metadata: Optional[str] = Form(None),  # JSON string
        image_url: Optional[str] = Form(None),
        image: Optional[UploadFile] = File(None),
    ):
        """Update object"""
        try:
            obj_uuid = uuid.UUID(object_id)

            # Parse metadata
            metadata_dict = None
            if metadata:
                import json

                metadata_dict = json.loads(metadata)

            # Decode image if provided
            img = None
            if image:
                image_data = await image.read()
                img = _decode_image(image_data)

            # Update object
            updated = manager.update(
                object_id=obj_uuid,
                name=name,
                metadata=metadata_dict,
                image_url=image_url,
                image=img,
            )

            if not updated:
                raise HTTPException(status_code=404, detail="Object not found")

            return {"status": "success", "id": object_id}

        except EmbeddingNotFoundError:
            raise HTTPException(status_code=404, detail="Object not found")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid UUID format")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    @app.post("/search")
    async def search(
        category: str = Form(...),
        embedding: Optional[str] = Form(None),  # JSON array or base64
        top_k: int = Form(10),
        similarity_threshold: float = Form(0.0),
        image: Optional[UploadFile] = File(None),
    ):
        """Search for similar objects"""
        import logging
        import traceback
        
        logger = logging.getLogger(__name__)
        
        try:
            import json

            # Get embedding from image or provided vector
            embedding_vec = None
            img = None
            if image:
                image_data = await image.read()
                img = _decode_image(image_data)
            elif embedding:
                # Parse embedding
                try:
                    embedding_vec = np.array(json.loads(embedding), dtype=np.float32)
                except json.JSONDecodeError:
                    # Try base64 decode
                    embedding_data = base64.b64decode(embedding)
                    embedding_vec = np.frombuffer(embedding_data, dtype=np.float32)
            else:
                raise HTTPException(status_code=400, detail="Either image or embedding must be provided")

            # Search
            results = manager.search(
                category=category,
                image=img if image else None,
                embedding=embedding_vec,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

            return {"results": results, "count": len(results)}

        except InvalidCategoryError as e:
            logger.error(f"InvalidCategoryError in /search: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except EmbeddingServiceError as e:
            logger.error(f"EmbeddingServiceError in /search: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error in /search: {e}", exc_info=True)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    @app.get("/categories")
    async def list_categories():
        """List all categories"""
        try:
            categories = manager.list_categories()
            counts = manager.count_by_category()

            return {
                "categories": categories,
                "counts": counts,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    @app.get("/categories/{category}/embeddings")
    async def get_category_embeddings(
        category: str,
        embedding_model: Optional[str] = None,
    ):
        """
        Get all embeddings for a category.
        
        This endpoint is useful for local similarity computation when you have
        frame embeddings and want to compare them with all category embeddings
        without making HTTP requests for each frame.
        
        Args:
            category: Category name (e.g., "franchise")
            embedding_model: Optional model name filter (e.g., "clip_224")
            
        Returns:
            List of embeddings with keys: id, category, name, embedding_model,
            embedding_dim, embedding (as list), metadata, image_url, added_at
        """
        try:
            embeddings = manager.get_all_embeddings(category, embedding_model)
            
            # Convert numpy arrays to lists for JSON serialization
            result = []
            for emb in embeddings:
                emb_dict = dict(emb)
                if isinstance(emb_dict.get("embedding"), np.ndarray):
                    emb_dict["embedding"] = emb_dict["embedding"].tolist()
                result.append(emb_dict)
            
            return {"embeddings": result, "count": len(result)}

        except InvalidCategoryError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except EmbeddingServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    @app.get("/categories/{category}/count")
    async def count_category(category: str):
        """Count objects in category"""
        try:
            counts = manager.count_by_category(category)
            return {"category": category, "count": counts.get(category, 0)}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")
        """
        Get all embeddings for a category.
        
        This endpoint is useful for local similarity computation when you have
        frame embeddings and want to compare them with all category embeddings
        without making HTTP requests for each frame.
        
        Args:
            category: Category name (e.g., "franchise")
            embedding_model: Optional model name filter (e.g., "clip_224")
            
        Returns:
            List of embeddings with keys: id, category, name, embedding_model,
            embedding_dim, embedding (as list), metadata, image_url, added_at
        """
        try:
            embeddings = manager.get_all_embeddings(category, embedding_model)
            
            # Convert numpy arrays to lists for JSON serialization
            result = []
            for emb in embeddings:
                emb_dict = dict(emb)
                if isinstance(emb_dict.get("embedding"), np.ndarray):
                    emb_dict["embedding"] = emb_dict["embedding"].tolist()
                result.append(emb_dict)
            
            return {"embeddings": result, "count": len(result)}

        except InvalidCategoryError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except EmbeddingServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    @app.post("/embed")
    async def embed(
        category: str = Form(...),
        image: UploadFile = File(...),
    ):
        """Extract embedding from image"""
        try:
            # Read image
            image_data = await image.read()
            img = _decode_image(image_data)

            # Get manager for category
            category_manager = manager._get_manager(category)

            # Extract embedding
            embedding = category_manager.extract_embedding(img)

            return {
                "embedding": embedding.tolist(),
                "dimension": len(embedding),
                "model": category_manager.model_name,
            }

        except InvalidCategoryError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except EmbeddingServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    @app.get("/health")
    async def health():
        """Health check"""
        return {"status": "healthy"}

    return app


# Global app instance for uvicorn (when using string import)
# This allows uvicorn to reload the app properly
app = create_app()


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    from ..config.settings import EmbeddingServiceConfig

    config = EmbeddingServiceConfig()
    app_instance = create_app(config)
    uvicorn.run(app_instance, host=config.server_host, port=config.server_port)

