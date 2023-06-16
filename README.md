# WBC Differential Learning Tool
Human peripheral blood contains 3 main types of cells, red blood cells, white blood cells and platelets. In normal clinical laboratory routine tests, we usually further differentiate white blood cells into 5 main different classes: **neutrophil, lymphocyte, basophil, eosinophil and monocyte**. 

The reason why we would like to classify those cells is because every class should have a set percentage of distribution in the normal human body. The sudden change in percentage of each cell class gives physicians an idea of what went wrong in the body. For example, abnormally high basophils and lymphocytes can mean you have chronic infection, cancer or other autoimmune disease. Sudden abnormal increase of eosinophil might mean your body is fighting for parasitic infection. Since the classification of these white blood cells affect the further treatment decision, it is very important that we correctly classify them.

## Data
The dataset that I use for this module is from a free public resource from Raabin-WBC data
Free Data - Raabin Health Database (raabindata.com). As I went through the dataset, I’ve noticed some of the pictures contain cells from different classes in the same picture, but they’ve only labeled the most centered cell. Therefore, I manually relabel them using Roboflow, which creates a useful dataset for the YOLOv5 model.
