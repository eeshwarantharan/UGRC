import logging
from src.perception import track_person_masks
from src.config import CAM_FILES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_sam_all")

def main():
    logger.info("Running SAM2 propagation for all cameras...")
    for cam in sorted(CAM_FILES.keys()):
        logger.info(f"Processing {cam}")
        try:
            track_person_masks(cam, force_recompute=True)
            logger.info(f"Done {cam}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"SAM failed for {cam}: {e}")

    logger.info("All cameras processed. Masks saved in: output/masks")

if __name__ == "__main__":
    main()

