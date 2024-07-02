//const { c } = require("vite/dist/node/types.d-aGj9QkWt");

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

// Load the GLTF model and add a ground plane for reference
const loader = new THREE.GLTFLoader();
let drone;
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

    // Add a ground plane for reference
    const geometry = new THREE.PlaneGeometry(100, 100);
    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.DoubleSide });
    const plane = new THREE.Mesh(geometry, material);
    plane.rotation.x = Math.PI / 2;  // Rotate to make it horizontal
    scene.add(plane);

    // Set camera starting position
    camera.position.z = 5.5;

    // WebSocket setup
    const socket = new WebSocket('ws://localhost:8081'); // Replace with your WebSocket server address

    let speed = 0.02;
    let moveInterval;
    let moveCommand;
    let droneFlying = false;
    let droneStartingPosition = drone.position;
    const maxheight = 4;
    const minheight = -5;
    const maxwidth = 9.5;
    const minwidth = -9.5;
    const maxdepth = 4;
    const mindepth = -7.5;

    socket.onopen = function () {
        console.log('WebSocket connection opened');
    };

    socket.onmessage = function (event) {
        console.log('WebSocket message received:', event.data);
        const command = JSON.parse(event.data);
        
        if (command === 'cut') {
            moveCommand = null;
            document.getElementById('current-movement').innerText = "None";
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

        // update the drone movement on the screen with the new movement
        document.getElementById('current-movement').innerText = moveCommand;

        if (droneFlying || moveCommand === 'clap') {
            moveInterval = setInterval(() => {
                
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
                    case 'clap':
                        if (drone.position == droneStartingPosition) {
                            takeOff = setInterval(() => {
                                if (drone.position.y >= 0) {
                                    clearInterval(takeOff);
                                    droneFlying = true;
                                }else {
                                    drone.position.y += speed;
                                }
                            }, 100);
                        } else {
                            // move the drone to the starting position
                            land = setInterval(() => {
                                if (drone.position == droneStartingPosition) {
                                    clearInterval(land);
                                    droneFlying = false;
                                }

                                if (drone.position.y != droneStartingPosition.y) {
                                    if (drone.position.y < droneStartingPosition.y) {
                                        drone.position.y += speed;
                                    } else {
                                        drone.position.y -= speed;
                                    }
                                }

                                if (drone.position.x != droneStartingPosition.x) {
                                    if (drone.position.x < droneStartingPosition.x) {
                                        drone.position.x += speed;
                                    } else {
                                        drone.position.x -= speed;
                                    }
                                }

                                if (drone.position.z != droneStartingPosition.z) {
                                    if (drone.position.z < droneStartingPosition.z) {
                                        drone.position.z += speed;
                                    } else {
                                        drone.position.z -= speed;
                                    }
                                }
                                
                            }, 100);
                        }
                        break;
                    default:
                        console.log('Unknown command:', moveCommand);
                }
                console.log('Drone position:', drone.position);
            }, 100); // Adjust interval time as needed
        }
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
