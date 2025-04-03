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
* **Ordering of running model:** Power efficiency, because the face recognition model uses the most power compared to the wake word model. since we have to consider real world usage, if the face recognition model is being ran first, before the mic, it will consume more power.


## limitations
* **face recognition:** if there are multiple faces in the frame being processed, it should show unrecognized, but currently, our system doesnt do this.
* **sound device driver:** the sounddevice library in python doesnt work well with edge devices
* **database:** currently our system only saves, 1 and only 1 face locally. we should have made a database system on cloud and allow for multiple faces stored to do face recognition.
