# ![alt text](./React/aidentity/public/shield.png) AiDentity

## Purpose

Implementing facial recognition technology can enhance security and surveillance by providing a robust and efficient means of identifying individuals in various settings. This technology allows for quick and accurate identification of persons of interest or unauthorised individuals. The goal of the system is to be able to detect and identify individuals accurately based on their facial features and structure.

This concept addresses possible border control scenarios with its purpose being to assist in remote locations with resource limitations that would benefit from an automated system. Other cases that would benefit from the use of such a system would be at airports with high passenger volumes

## Installation Guide
(Placeholder)
```
cd existing_repo
git remote add origin https://git.chalmers.se/courses/dit826/2023/group3/monorepo.git
git branch -M main
git push -uf origin main
```

## Description
The system’s functionality include an algorithm that identifies and recognizes a person's face in front of the camera in real-time through a user-friendly interface. This is done by creating and training a model based on a facial database in order to be able to identify that what is shown in the camera is a face.

The users would be airport security or border control personnel. The admins will be part of the developers. An example on who the program can be used on would be a person who's trying to cross the borders of one country to another for various reasons like tourism. Specifically, this said person would get their facial features analysed in order to determine if the person being scanned belongs to the database of national citizens of the country he wishes to enter. Our system can be used in airports or border entry points by the staff to inspect if any person entering the country is owing permanent residency to the country or not. If the person being inspected is flagged as not being included in the database of nationals, the system will inform the user that the person’s request for entry has been denied. Additional features are available for users with administrator access to the system. These features include: adding new data for the model to retrain on, viewing the previous models, and rolling back to different model versions if needed.

## Model versions

- **model_version_20231222020334.h5:** 
This h5 file contains the saved weights of the most recent trained VGG16 model.

- **model_version_20231229144933.h5:**
This h5 file contains the saved weights of a version of a trained EfficientNet model.

- **model_version_20240102024654.h5:**
This h5 file contains the saved weights of the most recent trained EfficientNet model.

- **model_version_20240102032316.h5:**
This h5 file contains the saved weights of a retrained version of EfficientNet model. This model version is the highest performing model in our system and its evaluation metrics can be seen in  /Model/model_registry/evaluation_metrics.json.

## Project Planning

This section includes everything related to our project planning. 

#### Project Prototype
![alt text](https://git.chalmers.se/courses/dit826/2023/group3/monorepo/-/design_management/designs/92/0a95da4f5825f67034460025aadd093e1f34a45b/raw_image)
![alt text](https://git.chalmers.se/courses/dit826/2023/group3/monorepo/-/design_management/designs/93/0a95da4f5825f67034460025aadd093e1f34a45b/raw_image)

#### Functional Decomposition Diagram
![alt text](https://git.chalmers.se/courses/dit826/2023/group3/monorepo/-/design_management/designs/106/792c0f742194435cd00724fd530673e8584b6e6f/raw_image)

#### Component Diagram
![alt text](https://git.chalmers.se/courses/dit826/2023/group3/monorepo/-/design_management/designs/107/617838b67b72f0b1d96fc0d670098d63773ccc46/raw_image)

## Authors
- Shahd Metwally (metwally)
- Sepehr Moradian (sepehrmo)
- Jennifer Hälgh (halgh)
- Sadhana Anandan (sadhana)
- Dimitrios Pokkias (pokkias)
