import os
import sys
import time

class DeepHeme():
    ### DeepHeme class, this is a class for automatic classigy the bone marrow cells on a whole slide images
    ### the input is a patches repo which is cropped from the whole slide images
    ### it has three major functions, each function will call a script to do the job
    ### the first script is a patch classification model, which will classify the patches into 3 classes:
    ### This is the command line: python Region_classifier.py --patch_repo_dir /media/hdd3/harry/Slides_repo/Plasma_cel_myeloma/patches/H18-2459_S10_MSKC_2023-05-31_16.29.27patches --save_vis False
    ### the second script is a cell detection model, which will detect the cells in the patches and save the coordinates of the cells
    ### This is the command line: python Cell_detector.py --patch_repo_dir /media/hdd3/harry/Slides_repo/Plasma_cel_myeloma/patches/H18-2459_S10_MSKC_2023-05-31_16.29.27patches/
    ### the third script is a cell classification model, which will classify the cells into a variety of classes
    ### This is the command line: !python Cell_classifier.py --patch_repo_dir /media/hdd3/harry/Slides_repo/Plasma_cel_myeloma/patches/H18-2459_S10_MSKC_2023-05-31_16.29.27patches 
    ### Please call the function and perform the analysis
    ### In the time also inform the user the progress of the analysis as well as the total time cost
    def __init__(self, registered = False, patch_repo_dir = None):
        self.registered = registered
        self.patch_repo_dir = patch_repo_dir
    
    def set_patch_repo_dir(self, patch_repo_dir):
        ### This is the function to set the patch repo directory
        self.patch_repo_dir = patch_repo_dir
        assert os.path.exists(self.patch_repo_dir), "The patch repo directory does not exist"
        self.registered = True
    
    def patch_classifier(self):
        ### This is the function to call the patch classifier
        ### This is the command line: python Region_classifier.py --patch_repo_dir /media/hdd3/harry/Slides_repo/Plasma_cel_myeloma/patches/H18-2459_S10_MSKC_2023-05-31_16.29.27patches --save_vis False
        start_time = time.time()
        print("Start patch classification")
        os.system("python Region_classifier.py --patch_repo_dir {} --save_vis False".format(self.patch_repo_dir))
        print("Patch classification finished")
        print("Time cost: {} seconds".format(time.time() - start_time))
    
    def cell_detector(self):
        ### This is the function to call the cell detector
        ### This is the command line: python Cell_detector.py --patch_repo_dir /media/hdd3/harry/Slides_repo/Plasma_cel_myeloma/patches/H18-2459_S10_MSKC_2023-05-31_16.29.27patches/
        start_time = time.time()
        print("Start cell detection")
        os.system("python Cell_detector.py --patch_repo_dir {}".format(self.patch_repo_dir))
        print("Cell detection finished")
        print("Time cost: {} seconds".format(time.time() - start_time))
    
    def cell_classifier(self):
        ### This is the function to call the cell classifier
        ### This is the command line: !python Cell_classifier.py --patch_repo_dir /media/hdd3/harry/Slides_repo/Plasma_cel_myeloma/patches/H18-2459_S10_MSKC_2023-05-31_16.29.27patches 
        start_time = time.time()
        print("Start cell classification")
        os.system("python Cell_classifier.py --patch_repo_dir {}".format(self.patch_repo_dir))
        print("Cell classification finished")
        print("Time cost: {} seconds".format(time.time() - start_time))

    def ready(self):
        ### check if the registration is done
        return self.registered

    def run(self):
        assert self.ready(), "Please register the patch repo directory first"
        ### This is the function to run the whole analysis
        self.patch_classifier()
        self.cell_detector()
        self.cell_classifier()
        print("DeepHeme analysis finished")
        print("Please check the results in the cell repo directory")

if __name__ == "__main__":
    ### example of how to use the DeepHeme class
    ### first register the patch repo directory
    ### then run the analysis
    ### the analysis will take a while, please be patient
    ### the results will be saved in the cell repo directory
    
    ### initialize the DeepHeme class
    deepheme = DeepHeme()
    ### register the patch repo directory
    deepheme.set_patch_repo_dir("/media/hdd3/harry/Slides_repo/Plasma_cel_myeloma/patches/H18-2459_S10_MSKC_2023-05-31_16.29.27patches")
    ### run the analysis
    deepheme.run()

    