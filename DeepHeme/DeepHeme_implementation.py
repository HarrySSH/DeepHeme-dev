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
    def __init__(self, registered = False,):
        self.registered = registered
        self.wsi_dir = None
        self.wsi_name = None
        self.patch_repo_dir = None
        #self.patch_repo_dir = patch_repo_dir
    
    def set_wsi_dir(self, wsi_dir, wsi_name):
        ### This is the function to set the wsi directory
        self.wsi_dir = wsi_dir
        assert os.path.exists(self.wsi_dir), "The wsi directory does not exist"
        self.wsi_name = wsi_name
        assert os.path.exists(os.path.join(self.wsi_dir, self.wsi_name)), "The wsi does not exist"
        ### created the overall patch foler format from /media/hdd4/harry/Slides_repo/AML/slides/ -> /media/hdd4/harry/Slides_repo/AML/patches/
        if not os.path.exists(self.wsi_dir.split('slides')[0] + 'patches'):
            os.mkdir(self.wsi_dir.split('slides')[0] + 'patches')

        self.patch_repo_dir = self.wsi_dir.split('slides')[0] + 'patches' + self.wsi_name.split('.ndpi')[0] + 'patches'
        if not os.path.exists(self.patch_repo_dir):
            os.mkdir(self.patch_repo_dir)
        self.registered = True
    
    def set_patch_repo_dir(self, patch_repo_dir):
        ### This is the function to set the patch repo directory
        self.patch_repo_dir = patch_repo_dir
        assert os.path.exists(self.patch_repo_dir), "The patch repo directory does not exist"
        self.registered = True

    def patching(self):
        ### this is the function to perforn patching in the multi-processing way
        start_time = time.time()
        print("Start patching")
        
        os.system("python mask_and_patch.py --wsi_dir {} --wsi_name {} ".format(self.wsi_dir, self.wsi_name))
        end_time = time.time()
        print("Patching finished")
        print("Time cost: {} seconds".format(end_time - start_time))
    
    def patch_classifier(self, ray = True):
        ### This is the function to call the patch classifier
        ### This is the command line: python Region_classifier.py --patch_repo_dir /media/hdd3/harry/Slides_repo/Plasma_cel_myeloma/patches/H18-2459_S10_MSKC_2023-05-31_16.29.27patches --save_vis False
        print("Start patch classification")
        start_time = time.time()
        if ray:
            os.system("python Ray_Region_classifier.py --patch_repo_dir {} ".format(self.patch_repo_dir))
        else:
            print('Using the non ray version')
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
        self.patching()
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

    