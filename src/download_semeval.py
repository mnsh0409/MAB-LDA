import os
import urllib.request
import time

# Define the exact raw GitHub URLs for the mirrored XML files
MIRROR_BASE = "https://raw.githubusercontent.com/alexcadillon/VicunaABSA/main/datasets/"

DATASETS = {
    "data/SemEval14/Restaurants_Train_v2.xml": MIRROR_BASE + "SemEval2014Task4/SemEval'14-ABSA-TrainData_v2%20%26%20AnnotationGuidelines/Restaurants_Train_v2.xml",
    "data/SemEval14/Restaurants_Test_Data_phaseB.xml": MIRROR_BASE + "SemEval2014Task4/ABSA_TestData_PhaseB/Restaurants_Test_Data_phaseB.xml",
    "data/SemEval14/Laptop_Train_v2.xml": MIRROR_BASE + "SemEval2014Task4/SemEval'14-ABSA-TrainData_v2%20%26%20AnnotationGuidelines/Laptop_Train_v2.xml",
    "data/SemEval14/Laptops_Test_Data_phaseB.xml": MIRROR_BASE + "SemEval2014Task4/ABSA_TestData_PhaseB/Laptops_Test_Data_phaseB.xml",
    
    "data/SemEval15/ABSA15_Restaurants_Train.xml": MIRROR_BASE + "SemEval2015Task12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml",
    "data/SemEval15/ABSA15_Restaurants_Test.xml": MIRROR_BASE + "SemEval2015Task12/ABSA15_Restaurants_Test.xml",
    
    "data/SemEval16/ABSA16_Restaurants_Train_SB1_v2.xml": MIRROR_BASE + "SemEval2016Task5/ABSA16_Restaurants_Train_SB1_v2.xml",
    "data/SemEval16/ABSA16_Restaurants_Test_SB1_v2.xml": MIRROR_BASE + "SemEval2016Task5/EN_REST_SB1_TEST.xml.gold"
}

def download_datasets():
    print("Starting SemEval XML downloads from academic mirror...")
    
    for local_path, url in DATASETS.items():
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        print(f"Fetching: {os.path.basename(local_path)}...")
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"  --> Saved to {local_path}")
        except Exception as e:
            print(f"  [!] Failed to download {local_path}: {e}")
        
        # Small delay to prevent hitting GitHub rate limits
        time.sleep(1)
        
    print("\nDownload complete! Your data/ directory is ready for run_all2_final.sh.")

if __name__ == "__main__":
    download_datasets()
