from src.listmonk import create_campaign, send_campaign
from src.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Monitor forecast trigger pipeline started.")
    campaign_id = create_campaign()
    logger.info(f"Created campaign with ID: {campaign_id}")
    send_campaign(campaign_id)
    logger.info("Monitor forecast trigger pipeline completed.")
