from fastapi import APIRouter
import requests
import cv2
from schemas.schema import ImageURL
from services.roof_damage_detection_services import RoofDamageDetectionServices as rds


router = APIRouter(
    prefix='/automation'
)


@router.post('/roofDamageDetection')
async def roof_damage_detection(image_url: ImageURL):
    #image_path = requests.get((unquote(image_url.image)))
    image = cv2.imread(image_url.image)
    response = rds.roof_or_not_detection(image)
    return response