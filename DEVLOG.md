# Things we tried

## Face Recognition

* **MobileNetV2 (70%):** We experimented with MobileNetV2 at a 70% compression rate for face recognition. However, we found that the similarity scores were too close, leading to unreliable recognition results.
* **ResNet50 (Custom Training):** To improve accuracy, we trained a ResNet50 model using a dataset from Kaggle. This approach yielded better recognition performance compared to MobileNetV2.

## Wake Word Model

* **CNN Architectures:** We explored various Convolutional Neural Network (CNN) architectures for our wake word detection model.
    * **5-Layer CNN:** Achieved an accuracy of approximately 99.29%.
    * **4-Layer CNN:** Also achieved an accuracy of approximately 99.42%.
    * **3-Layer CNN:** We settled on a 3-layer CNN. While it had a slightly lower accuracy (around 98%), it demonstrated better voice isolation, which was crucial for reliable wake word detection in noisy environments.

* **Architecture Iterations:** We made specific changes to the 3-layer CNN architecture during development:

    | Element     | First Version                 | Second Version             |
    |-------------|-------------------------------|----------------------------|
    | Dropout     | Yes ( `nn.Dropout(0.5)` )     | **X** Not included         |
    | BatchNorm   | Yes after each Conv layer    | **X** Not included         |
    | Sigmoid     | Not in `forward()` output      | ✅ Present in final layer  |
    | Loss        | `BCEWithLogitsLoss()`         | `BCELoss()`                |

    These modifications were made to optimize performance, reduce complexity, and improve the model's ability to isolate the wake word from background noise. We found that removing Dropout and BatchNorm reduced the model's complexity and improved training stability in our specific use case. The inclusion of the Sigmoid function in the final layer, coupled with the change in the loss function, resulted in better classification performance for our wake word detection task.

## User Interface

* **Web-Based UIs (Streamlit/Flask):** We initially explored using web-based user interfaces like Streamlit and Flask. This approach would have allowed us to leverage the processing power of our PCs instead of relying solely on the Raspberry Pi. However, we encountered concerns regarding compatibility with edge devices.
* **Tkinter:** Ultimately, we opted for Tkinter for the user interface. Tkinter provides a simple and reliable GUI framework that is compatible with the Raspberry Pi and other edge devices, ensuring consistent performance.
    * **Threading Implementation:** Due to the need to run certain processes concurrently (like live camera feed processing, audio analysis, and sensor monitoring) without freezing the UI, we had to implement threading. This allowed us to maintain a responsive user interface while background tasks were being executed. This was especially important for the live camera display and audio analysis which would otherwise cause the UI to become unresponsive.
 
## Application Jusitification
* **Ordering of running model:** The face recognition model, due to its higher computational intensity, consumes significantly more power than the wake word detection model. To minimize overall energy usage in real-world scenarios, the system prioritizes executing the face recognition model first. This strategy ensures that power-intensive processing is completed before the microphone is activated, leading to a more efficient power profile.


## Limitations
* **Multi-Face Ambiguity in Face Recognition:** The current implementation lacks robust handling of scenarios with multiple faces within the camera's field of view. Instead of accurately identifying and flagging such instances as "unrecognized" due to ambiguity, the system may produce unpredictable or erroneous results.
* **Edge Device Compatibility with Sound Device Driver:** The sounddevice Python library exhibits compatibility issues with edge devices, potentially leading to unreliable audio capture and processing. This limitation may hinder the system's performance in real-world deployments on resource-constrained hardware.
* **Scalability and Multi-User Support via Local Database Constraint:** The system's reliance on a local database for face storage restricts its scalability and multi-user capabilities. A cloud-based database architecture should be implemented to enable the storage and management of multiple user profiles, facilitating robust and scalable face recognition functionality.
