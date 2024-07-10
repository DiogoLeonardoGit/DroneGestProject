// Set up the scene, camera, and renderer
const scene = new THREE.Scene();

// Load a background image
const textureLoader = new THREE.TextureLoader();
textureLoader.load('textures/background2.jpg', function (texture) {
    const bgGeometry = new THREE.PlaneGeometry(60, 30); // Adjust the size as needed
    const bgMaterial = new THREE.MeshBasicMaterial({ map: texture });
    const bgMesh = new THREE.Mesh(bgGeometry, bgMaterial);

    // Position the background plane
    bgMesh.position.z = -10; // Move it back behind all other objects
    scene.add(bgMesh);
}, undefined, function (error) {
    console.error('Error loading background image:', error);
});

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Adjust the renderer and camera on window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// Add lighting
const light = new THREE.AmbientLight(0x808080); // Stronger ambient light
scene.add(light);
const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
scene.add(directionalLight);

// Audio listener
const listener = new THREE.AudioListener();
camera.add(listener);

// Load the GLTF model and add a ground plane for reference
const loader = new THREE.GLTFLoader();
let drone;
let sound;
let currentSound = 'takeoff';
loader.load('drone-model.glb', function (gltf) {
    drone = gltf.scene;
    scene.add(drone);
    drone.position.set(0, -3, 0);  // Position the drone at the origin
    drone.scale.set(1, 1, 1);  // Scale the drone if necessary

    // Load and apply texture to the drone
    const droneTexture = new THREE.TextureLoader().load('textures/droneTex.png'); // Replace with your texture path
    drone.traverse((node) => {
        if (node.isMesh) {
            node.castShadow = true;
            node.receiveShadow = true;
            node.material = new THREE.MeshStandardMaterial({ map: droneTexture }); // Apply the texture
        }
    });


    // Add audio to the drone
    const audioLoader = new THREE.AudioLoader();
    const takeoffSound = new THREE.PositionalAudio(listener);
    const flyingSound = new THREE.PositionalAudio(listener);
    const landingSound = new THREE.PositionalAudio(listener);

    // Load sounds into respective audio instances
    audioLoader.load('sounds/takeoff.mp3', function (buffer) {
        takeoffSound.setBuffer(buffer);
        takeoffSound.setVolume(1);
        takeoffSound.setRefDistance(20);
    });
    audioLoader.load('sounds/flying.mp3', function (buffer) {
        flyingSound.setBuffer(buffer);
        flyingSound.setVolume(20); 
        flyingSound.setLoop(true);
        flyingSound.setRefDistance(20);
    });
    audioLoader.load('sounds/landing.mp3', function (buffer) {
        landingSound.setBuffer(buffer);
        landingSound.setVolume(1); 
        landingSound.setRefDistance(20);
    });

    // Add sounds to the drone
    drone.add(takeoffSound);
    drone.add(flyingSound);
    drone.add(landingSound);

    function switchSound(toPlay, toStop1, toStop2) {
        //const duration = 0.1;
        //const initialVolume = 1;

        // Start the new sound with volume 0 and fade it in
        toPlay.play();

        // Stop the other sounds
        if (toStop1.isPlaying) {
            toStop1.stop();
        }

        if (toStop2.isPlaying) {
            toStop2.stop();
        }
        /*
        // Fade out the sounds that need to be stopped
        const fadeOutInterval = setInterval(() => {
            if (toStop1 && toStop1.isPlaying && toStop1.getVolume() > 0) {
                toStop1.setVolume(toStop1.getVolume() - initialVolume / (duration * 100));
            } else if (toStop1 && toStop1.isPlaying) {
                toStop1.stop();
                toStop1.setVolume(initialVolume);
                clearInterval(fadeOutInterval);
            }

            if (toStop2 && toStop2.isPlaying && toStop2.getVolume() > 0) {
                toStop2.setVolume(toStop2.getVolume() - initialVolume / (duration * 100));
            } else if (toStop2 && toStop2.isPlaying) {
                toStop2.stop();
                toStop2.setVolume(initialVolume);
                clearInterval(fadeOutInterval);
            }
        }, 10);
        */
    }

    function changeSoundTo(newSound) {
        
        if (newSound === 'takeoff' && !takeoffSound.isPlaying) {
            switchSound(takeoffSound, flyingSound, landingSound);
            console.log('Changing sound to takeoff');
        } else if (newSound === 'flying' && !flyingSound.isPlaying) {
            switchSound(flyingSound, takeoffSound, landingSound);
            console.log('Changing sound to flying');
        } else if (newSound === 'landing' && !landingSound.isPlaying) {
            switchSound(landingSound, takeoffSound, flyingSound);
            console.log('Changing sound to landing');
        }
    }

    // Add a ground plane for reference
    const geometry = new THREE.PlaneGeometry(100, 100);
    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.DoubleSide });
    const plane = new THREE.Mesh(geometry, material);
    plane.rotation.x = Math.PI / 2;  // Rotate to make it horizontal
    scene.add(plane);

    // Set camera starting position
    camera.position.z = 5.5;

    // WebSocket setup
    const socket = new WebSocket('ws://localhost:8081'); // same as server

    // Parameters definition 
    let speed = 0.03;
    let moveInterval;
    let moveCommand;
    let droneFlying = false;
    let droneStartingPosition = new THREE.Vector3(0, -3, 0);  // set the initial drone position
    const maxheight = 4;
    const minheight = -4;
    const maxwidth = 8;
    const minwidth = -8;
    const maxdepth = 4;
    const mindepth = -7.5;

    // set label for the current movement
    document.getElementById('current-movement').innerText = "Landed";

    socket.onopen = function () {
        console.log('WebSocket connection opened');
    };

    socket.onmessage = function (event) {
        console.log('WebSocket message received:', event.data);
        const command = JSON.parse(event.data);

        if (command === 'cut') {
            moveCommand = null;
            document.getElementById('current-movement').innerText = "hovering";
            if (moveInterval) {
                clearInterval(moveInterval);
            }
            return;
        }

        if (command === 'noise' || command === 'unknown') {
            return;
        }

        if (command === 'recording') {
            // update recording time on the screen starting on 2.00 seconds to 0.00 seconds decreasing 0.01 seconds
            let recordingTime = 2.00;
            document.getElementById('recording-time').style.color = 'red';
            const recordingInterval = setInterval(() => {
                recordingTime -= 0.01;
                document.getElementById('recording-time').innerText = recordingTime.toFixed(2);
                if (recordingTime < 0.01) {
                    clearInterval(recordingInterval);
                    document.getElementById('recording-time').innerText = '0.00';
                    document.getElementById('recording-time').style.color = 'white';
                }
            }, 10); // Adjust interval time as needed
            return;
        }

        moveCommand = command;

        if (moveCommand === 'clap') {
            // Update the takeoff and landing code to use the new switchSound function
            if (!droneFlying) {
                // play the sound
                changeSoundTo('takeoff');
                takeOff = setInterval(() => {
                    if (drone.position.y >= 0) {
                        moveCommand = null;
                        document.getElementById('current-movement').innerText = "hovering";
                        if (takeOff) {
                            clearInterval(takeOff);
                        }
                        droneFlying = true;
                        if (currentSound !== 'flying') {
                            changeSoundTo('flying');
                            currentSound = 'flying';
                        }
                        return;
                    } else {
                        console.log('Taking off:', drone.position);
                        document.getElementById('current-movement').innerText = "Taking off";
                        setTimeout(() => {
                            drone.position.y += speed;
                        }, 400);
                    }
                }, 40);
            } else {
                // move the drone to the starting position
                if (currentSound !== 'landing') {
                    changeSoundTo('landing');
                    currentSound = 'landing';
                }
                land = setInterval(() => {
                    if (Math.abs(drone.position.y - droneStartingPosition.y) < 0.05 && Math.abs(drone.position.x - droneStartingPosition.x) < 0.05 && Math.abs(drone.position.z - droneStartingPosition.z) < 0.05) {
                        console.log('Landed:', drone.position);
                        moveCommand = null;
                        document.getElementById('current-movement').innerText = "Landed";
                        if (land) {
                            clearInterval(land);
                        }
                        clearInterval(moveInterval);
                        droneFlying = false;
                        landingSound.setLoop(false);
                        return;
                    } else {
                        document.getElementById('current-movement').innerText = "Landing";
                        console.log('Landing:', drone.position);
                        if (Math.abs(drone.position.y - droneStartingPosition.y) > 0.05) {
                            if (drone.position.y < droneStartingPosition.y) {
                                drone.position.y += speed;
                            } else {
                                drone.position.y -= speed;
                            }
                        }

                        if (Math.abs(drone.position.x - droneStartingPosition.x) > 0.05) {
                            if (drone.position.x < droneStartingPosition.x) {
                                drone.position.x += speed;
                            } else {
                                drone.position.x -= speed;
                            }
                        }

                        if (Math.abs(drone.position.z - droneStartingPosition.z) > 0.05) {
                            if (drone.position.z < droneStartingPosition.z) {
                                drone.position.z += speed;
                            } else {
                                drone.position.z -= speed;
                            }
                        }
                    }
                }, 35);
            }
            return;
        }

        // update the drone movement on the screen with the new movement
        document.getElementById('current-movement').innerText = moveCommand;

        if (droneFlying) {

            if (moveInterval) {
                clearInterval(moveInterval);
            }

            moveInterval = setInterval(() => {
                speed = 0.03;
                switch (moveCommand) {
                    case 'up':
                        if (drone.position.y >= maxheight) {
                            break;
                        } else {
                            drone.position.y += speed;
                            break;
                        }
                    case 'down':
                        if (drone.position.y <= minheight) {
                            break;
                        } else {
                            drone.position.y -= speed;
                            break;
                        }
                    case 'left':
                        if (drone.position.x <= minwidth) {
                            break;
                        } else {
                            drone.position.x -= speed;
                            break;
                        }
                    case 'right':
                        if (drone.position.x >= maxwidth) {
                            break;
                        } else {
                            drone.position.x += speed;
                            break;
                        }
                    case 'front':
                        if (drone.position.z >= maxdepth) {
                            break;
                        } else {
                            drone.position.z += speed;
                            break;
                        }
                    case 'back':
                        if (drone.position.z <= mindepth) {
                            break;
                        } else {
                            drone.position.z -= speed;
                            break;
                        }
                    case 'spin':
                        drone.rotation.y += speed;
                        break;
                    default:
                        console.log('Unknown command:', moveCommand);
                }
            }, 50);
        }
        return;
    };

    socket.onclose = function () {
        console.log('WebSocket connection closed');
        if (moveInterval) {
            clearInterval(moveInterval);
        }
    };

    socket.onerror = function (error) {
        console.error('WebSocket error:', error);
        if (moveInterval) {
            clearInterval(moveInterval);
        }
    };

    // Add event listener for mouse clicks
    window.addEventListener('mousedown', function () {
        if (socket.readyState === WebSocket.OPEN) {
            const message = JSON.stringify({ event: 'mouseClick' });
            socket.send(message);
            console.log('Mouse click event sent to server');
        }
    });

    // Render loop
    function animate() {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    }

    animate(); // Start the render loop after setting up everything
}, undefined, function (error) {
    console.error('Error loading drone model:', error);
});

// Add zoom in/out functionality
window.addEventListener('wheel', (event) => {
    const zoomSpeed = 1.1;

    // define min and max zoom
    const minZoom = 7.5; // starting zoom
    const maxZoom = 0.2; // maximum zoom

    if (event.deltaY < 0) {
        if (camera.position.z <= maxZoom) {
            return;
        }
        camera.position.z /= zoomSpeed;
    } else {
        if (camera.position.z >= minZoom) {
            return;
        }
        camera.position.z *= zoomSpeed;
    }

    console.log('Zoom:', camera.position.z);
});
