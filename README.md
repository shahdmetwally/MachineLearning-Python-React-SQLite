# AiDenity

## Purpose

This project's purpose is to make it quick and easy for travelers to get their identity verified using facial recognition.The goal of the system is to be able to detect and identify individuals accurately based on their facial features and structure.

This concept addresses possible border control scenarios' purpose being to assist in locations with resource limitations or understaffed and remote locations that would benefit from an automated system. Other cases that would benefit from the use of such a system would be at airports with high passenger volumes.

## Installation Guide
(Placeholder)
```
cd existing_repo
git remote add origin https://git.chalmers.se/courses/dit826/2023/group3/monorepo.git
git branch -M main
git push -uf origin main
```

## Description
The system’s functionality will include an algorithm that identifies and recognizes a person's face in front of the camera in real-time through a user-friendly interface. This will be done by creating and training a model based on a facial database in order to be able to identify that what is shown in the camera is a face.
The users we will have will be normal users and admins. An ordinary user can be someone who’s trying to cross the borders of one country to another for various reasons like tourism. Specifically, this user would get their facial features analysed in order to determine if the person being scanned belongs to the database of national citizens of the country he wishes to enter. Our system can be used in airports or border entry points to inspect if the person entering the country is owing permanent residency to the country or not. If the person being inspected is flagged as not being included in the database of nationals, the system will inform that person that their request for entry has been denied and that they should seek a staff member for further guidance. Additional features are available for users with administrator access to the system, like for example adding new data to possibly train the model on, or update it.

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
![alt text](https://git.chalmers.se/courses/dit826/2023/group3/monorepo/-/design_management/designs/103/70d68b399fb23cfeef6ce9de2c72ed28dfa605b9/raw_image)

#### Component Diagram
![alt text](https://git.chalmers.se/courses/dit826/2023/group3/monorepo/-/design_management/designs/99/17210d08df328b1c7b62cba70eda5795afa5c6b5/raw_image)

## Authors
- Shahd Metwally (metwally)
- Sepehr Moradian (sepehrmo)
- Jennifer Hälgh (halgh)
- Sadhana Anandan (sadhana)
- Dimitrios Pokkias (pokkias)
