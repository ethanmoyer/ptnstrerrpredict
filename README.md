# ptnstrerrpredict - Determining an Energy Cost Function to Predict 3D Protein Structure using Machine Learning

## Install requirements
```
pip install -r requirements.txt
```

## Installing pyrosetta

http://www.pyrosetta.org/dow

## Program

This project is a part of my undergraduate curriculum at Drexel University in Philadelphia, PA. During the summer after my freshman year, I was selected as a STAR (Students Tackling Advanced Research) Scholar Research Assistant under Drexel University Office of Undergraduate Research. 

The STAR Scholars Program is a competitive research program that allows first-year students to participate in faculty-mentored research, scholarship, or creative work during the summer after their freshman year. The program provides an opportunity for students to get to know faculty, explore a major area of research, and gain practical skills and valuable research experience. It is designed to bring the brightest and most forward-thinking minds at Drexel together to independently develop novel research and creative ideas.

## Background

Over the course of my STAR experience, I will be focusing on the protein folding problem, which is a long-standing research topic in bioinformatics. In short, the nature of the problem is being able to predict the three-dimensional (3-D) structure of a protein given it primary amino acid sequence. Many methods have been employed to directly build, or fold, the 3-D structure of a protein based on its sequence. In my project, however, I will simply be creating a model to predict how much one proposed 3-D structure deviates from the protein’s most optimal structure. Using an energy cost function, along with selected factors, or features, of protein structure and primary sequence data, I plan to research and build a cost function that will be used to predict magnitude of this deviation. 

In the field of molecular biology, specifically in proteomics (the study of proteins), the “structure dictates function” paradigm generalizes that molecules with distinct structural features often have specific function in cells. Thus, the obvious application to this bioinformatics project is predicting protein structure and function based on the primary sequences of novel proteins. This generalized application can have far-reaching implications for the field of molecular biology and for big pharma—the creation of novel proteins with known structure and function can allow for specific disease treatment solutions and for specific genetically engineering novel biological pathways. Such advancements can be observed in international research competitions such as the International Genetically Engineered Machine Foundation. Research in this area is more concerned with the classification of proteins based on primary structure. Such a task is more difficult when the way in which a structure is built, or the characteristic of a specific protein function, is more or less unknown. 

