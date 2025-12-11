import os

from src.listmonk import create_campaign, send_campaign
from src.logger import get_logger

logger = get_logger(__name__)

TEST_EMAIL = os.getenv("TEST_EMAIL", "True")
FORCE_TRIGGER = os.getenv("FORCE_TRIGGER", "False")

if __name__ == "__main__":
    logger.info("Monitor forecast trigger pipeline started.")
    logger.info(f"TEST_EMAIL: {TEST_EMAIL}")
    logger.info(f"FORCE_TRIGGER: {FORCE_TRIGGER}")
    campaign_id = create_campaign()
    logger.info(f"Created campaign with ID: {campaign_id}")
    send_campaign(campaign_id)
    logger.info("Monitor forecast trigger pipeline completed.")
