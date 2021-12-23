$(document).ready(function () {

    async function buildFeatureLibrary() {
        const labels = ['Sinter', 'Man', 'Woman', 'Black Widow', 'Captain America', 'Captain Marvel', 'Hawkeye', 'Jim Rhodes', 'Thor', 'Tony Stark']
        return Promise.all(
            labels.map(async label => {
                const descriptions = []
                for (let i = 1; i <= 2; i++) {
                    const img = await faceapi.fetchImage("labeled_images/" + label + "/" + i + ".jpg")
                    const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
                    // const detections = await faceapi.detectSingleFace(img, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptor()
                    descriptions.push(detections.descriptor)
                }
                return new faceapi.LabeledFaceDescriptors(label, descriptions)
            })
        )
    }

    class Inference {
        constructor(video, info) {
            this.defaultInfo = {
                facingMode: "user",
                width: 640,
                height: 480
            };
            this.video = video;
            this.elapsed_time = 0;
            this.time = 0;
            this.info = Object.assign(Object.assign({}, this.defaultInfo), info);
        }

        run(stream) {
            this.video.srcObject = stream;
            this.video.onloadedmetadata = () => {
                this.video.play();
                this.captureStatus()
            }
        }

        captureStatus() {
            this.time++;  // 记录每一帧
            window.requestAnimationFrame(() => {
                this.forward()
            })
        }

        forward() {
            // 人脸识别
            if (this.time % 500 === 0) {
                var faceRecInfer = null;
                this.video.paused || this.video.currentTime === this.elapsed_time || (this.elapsed_time = this.video.currentTime, faceRecInfer = this.info.faceRecInfer());
                faceRecInfer ? faceRecInfer.then(() => {
                    this.captureStatus()
                }) : this.captureStatus()
            }
            if (this.time % 100 === 0) {
                var faceInfer = null;
                this.video.paused || this.video.currentTime === this.elapsed_time || (this.elapsed_time = this.video.currentTime, faceInfer = this.info.faceInfer());
                faceInfer ? faceInfer.then(() => {
                    this.captureStatus()
                }) : this.captureStatus()
            }
            else if (this.time % 20 === 0) {
                var handInfer = null;
                this.video.paused || this.video.currentTime === this.elapsed_time || (this.elapsed_time = this.video.currentTime, handInfer = this.info.handPointInfer());
                handInfer ? handInfer.then(() => {
                    this.captureStatus()
                }) : this.captureStatus()
            }
            else {
                this.captureStatus()
            }
        }

        start() {
            navigator.mediaDevices && navigator.mediaDevices.getUserMedia || alert("No navigator.mediaDevices.getUserMedia exists.");
            return navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: this.info.facingMode,
                    width: this.info.width,
                    height: this.info.height
                }
            }).then((stream) => {
                this.run(stream)
            }).catch((c) => {
                console.error("Failed to acquire camera feed: " + c);
                alert("Failed to acquire camera feed: " + c);
                throw c;
            })
        }
    }

    function calcAngleDegrees(x, y) {
        return Math.atan2(y, x) * 180 / Math.PI;
    }

    function isOpen(wrist, points, threshold) {
        // 手腕到手指中心为一个向量，手指第二段到指尖为一个向量，计算两个向量的角度
        let v1 = wrist.concat(points[1]);
        let v2 = points[2].concat(points[3]);

        dx1 = v1[2] - v1[0];
        dy1 = v1[3] - v1[1];
        dx2 = v2[2] - v2[0];
        dy2 = v2[3] - v2[1];

        const angle1 = calcAngleDegrees(dy1, dx1)

        const angle2 = calcAngleDegrees(dy2, dx2)

        if (angle1 * angle2 >= 0) {
            included_angle = Math.abs(angle1 - angle2)
        }
        else {
            included_angle = Math.abs(angle1) + Math.abs(angle2)
            if (included_angle > 180) {
                included_angle = 360 - included_angle
            }
        }
        if (included_angle < 90) {
            return "1";
        }
        else {
            return "0";
        }
    }

    function handsResults(results) {
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        // canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
        if (results.multiHandLandmarks) {
            for (const landmarks of results.multiHandLandmarks) {
                drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
                    { color: '#00FF00', lineWidth: 5 });
                drawLandmarks(canvasCtx, landmarks, { color: '#FF0000', lineWidth: 2 });
            }
        }
        results.multiHandLandmarks.forEach((landmarks, i) => {
            // 还原绝对坐标
            realLandmarks = [];
            for (let index = 0; index < landmarks.length; index++) {
                const element = landmarks[index];
                realLandmarks.push([element["x"] * canvasElement.width, element["y"] * canvasElement.height]);
            }

            wrist = realLandmarks[0];
            fingerStatus = []
            for (let index = 1; index < realLandmarks.length; index += 4) {
                fingerPoints = realLandmarks.slice(index, index + 4);
                fingerStatus.push(isOpen(wrist, fingerPoints, 90));
            }

            const actionKey = fingerStatus.join('')
            const action = handAction[actionKey];
            gestureElement.innerHTML = action;
        })
        // $.post("/hands", {
        //     pose_landmarks: JSON.stringify(results.multiHandLandmarks),
        //     width: canvasElement.width,
        //     height: canvasElement.height
        // }, function (err, req, resp) {
        //     console.log(resp);
        // });
    }

    function faceResults(detections) {
        canvasElement.getContext('2d').clearRect(0, 0, canvasElement.width, canvasElement.height)
        faceapi.draw.drawDetections(canvasElement, detections)
        faceapi.draw.drawFaceExpressions(canvasElement, detections)

        // $.post("/faces", {
        //     detections: JSON.stringify(detections),
        //     width: canvasElement.width,
        //     height: canvasElement.height
        // }, function (err, req, resp) {
        //     console.log(resp);
        // });
    }

    function faceRecResults(detections) {
        canvasElement.getContext('2d').clearRect(0, 0, canvasElement.width, canvasElement.height)
        const results = detections.map(d => faceMatcher.findBestMatch(d.descriptor))
        results.forEach((result, i) => {
            const box = detections[i].detection.box
            const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
            drawBox.draw(canvasElement)
        })
        // $.post("/faces_recognition", {
        //     detections: JSON.stringify(result),
        // }, function (err, req, resp) {
        //     console.log(resp);
        // });
    }

    // const videoElement = document.getElementsByClassName("input_video")[0];
    const videoElement = document.getElementById("video");
    const gestureElement = document.getElementById("gesture");
    const canvasElement = document.getElementsByClassName("output_canvas")[0];
    const canvasCtx = canvasElement.getContext("2d");

    self.handAction = {
        "00000": "fist",
        "01000": "one",
        "00100": "fuck",
        "11001": "love you",
        "01100": "Yeah",
        "01110": "three",
        "01111": "four",
        "11111": "palm",
        "00111": "OK",
        "00001": "low",
        "11000": "gun",
        "10001": "six",
        "10000": "thumb up"
    }

    const hands = new Hands({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
    });
    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    hands.onResults(handsResults)

    let faceMatcher = null;
    Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
        faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
        faceapi.nets.faceExpressionNet.loadFromUri('/models')
    ]).then(async () => {
        const labeledFaceDescriptors = await buildFeatureLibrary()
        faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
    })

    const displaySize = { width: videoElement.width, height: videoElement.height }

    const inference = new Inference(videoElement, {
        handPointInfer: async () => {
            await hands.send({ image: videoElement });
        },
        faceRecInfer: async () => {
            const detections = await faceapi.detectAllFaces(videoElement, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors()
            const resizedDetections = faceapi.resizeResults(detections, displaySize)
            faceRecResults(resizedDetections)
        },
        faceInfer: async () => {
            const detections = await faceapi.detectAllFaces(videoElement, new faceapi.TinyFaceDetectorOptions()).withFaceExpressions()
            const resizedDetections = faceapi.resizeResults(detections, displaySize)
            faceResults(resizedDetections)
        },
        width: 720,
        height: 560
    });
    inference.start();

});
