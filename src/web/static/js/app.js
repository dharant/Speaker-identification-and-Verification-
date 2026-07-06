/* JavaScript Application Logic - AuraDiarize Premium Interface */

document.addEventListener('DOMContentLoaded', () => {
    
    // UI Elements
    const dropZone = document.getElementById('drop-zone');
    const audioInput = document.getElementById('audio-input');
    const selectedFileName = document.getElementById('selected-file-name');
    const selectedFileSize = document.getElementById('selected-file-size');
    const dropZoneContent = document.querySelector('.drop-zone-content');
    const fileDetails = document.getElementById('file-details');
    const uploadForm = document.getElementById('upload-form');
    
    const welcomeState = document.getElementById('welcome-state');
    const loaderState = document.getElementById('loader-state');
    const resultsState = document.getElementById('results-state');
    
    const loaderTitle = document.getElementById('loader-title');
    const loaderDesc = document.getElementById('loader-desc');
    const progressBarFill = document.getElementById('progress-bar-fill');
    
    const resultsFilename = document.getElementById('results-filename');
    const resultsDuration = document.getElementById('results-duration');
    const resultsSpeakers = document.getElementById('results-speakers');
    
    const playBtn = document.getElementById('play-btn');
    const stopBtn = document.getElementById('stop-btn');
    const currentTimeDisplay = document.getElementById('current-time');
    const totalTimeDisplay = document.getElementById('total-time');
    const volumeSlider = document.getElementById('volume-slider');
    
    const speakerLegend = document.getElementById('speaker-legend');
    const timelineRibbon = document.getElementById('timeline-ribbon');
    const chatList = document.getElementById('chat-list');
    const exportBtn = document.getElementById('export-btn');
    
    // Application States
    let wavesurfer = null;
    let audioFileObject = null;
    let diarizationData = null;
    let speakerNamesMap = {}; // Maps "Speaker A" -> Custom name
    
    // Drag & Drop listeners
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('hover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('hover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            audioInput.files = files;
            handleFileSelection(files[0]);
        }
    });

    audioInput.addEventListener('change', (e) => {
        if (audioInput.files.length > 0) {
            handleFileSelection(audioInput.files[0]);
        }
    });

    function handleFileSelection(file) {
        audioFileObject = file;
        selectedFileName.textContent = file.name;
        selectedFileSize.textContent = formatBytes(file.size);
        
        dropZoneContent.classList.add('hidden');
        fileDetails.style.display = 'flex';
    }

    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    // Form Submission (Process Audio)
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!audioFileObject && audioInput.files.length > 0) {
            audioFileObject = audioInput.files[0];
        }
        
        if (!audioFileObject) {
            alert("Please select or drag-and-drop an audio file first.");
            return;
        }

        // Switch state
        welcomeState.classList.add('hidden');
        resultsState.classList.add('hidden');
        loaderState.classList.remove('hidden');
        
        // Start Progress Loader simulation
        let progress = 0;
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += Math.floor(Math.random() * 8) + 2;
                updateLoaderProgress(progress);
            }
        }, 300);

        // Build FormData
        const formData = new FormData();
        formData.append('audio', audioFileObject);
        formData.append('num_speakers', document.getElementById('num-speakers').value);
        formData.append('model_type', document.getElementById('model-type').value);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Server processing error');
            }

            const data = await response.json();
            diarizationData = data;
            
            // Finish Progress Bar
            updateLoaderProgress(100);
            setTimeout(() => {
                loaderState.classList.add('hidden');
                displayResults(data);
            }, 600);

        } catch (error) {
            clearInterval(progressInterval);
            loaderState.classList.add('hidden');
            welcomeState.classList.remove('hidden');
            alert(`Error processing audio: ${error.message}`);
        }
    });

    // Update progress details in UI
    function updateLoaderProgress(progress) {
        progressBarFill.style.width = `${progress}%`;
        
        const vad = document.getElementById('prog-vad');
        const embed = document.getElementById('prog-embed');
        const cluster = document.getElementById('prog-cluster');
        const asr = document.getElementById('prog-asr');

        if (progress < 25) {
            loaderTitle.textContent = "Segmenting Audio Waveform";
            loaderDesc.textContent = "Voice Activity Detection: Splitting dialogue turns...";
            vad.classList.add('active');
        } else if (progress < 50) {
            const modelVal = document.getElementById('model-type').value;
            loaderTitle.textContent = "Extracting Speaker Characteristics";
            let modelLabel = "Custom Model";
            if (modelVal === 'pretrained_ecapa') modelLabel = "SpeechBrain ECAPA-TDNN";
            else if (modelVal === 'pretrained_resnet') modelLabel = "SpeechBrain ResNet-34";
            else if (modelVal === 'pretrained_wavlm') modelLabel = "Microsoft WavLM (SOTA)";
            
            loaderDesc.textContent = `${modelLabel}: Generating speaker embeddings...`;
            embed.classList.add('active');
        } else if (progress < 75) {
            loaderTitle.textContent = "Grouping Speaker Vectors";
            loaderDesc.textContent = "Agglomerative Clustering: Matching voices across segments...";
            cluster.classList.add('active');
        } else {
            loaderTitle.textContent = "Transcribing Dialogue Segments";
            loaderDesc.textContent = "ASR System: Translating voices to text...";
            asr.classList.add('active');
        }
    }

    // Display Results in Dashboard
    function displayResults(data) {
        resultsFilename.textContent = data.filename;
        resultsDuration.textContent = data.duration;
        resultsSpeakers.textContent = data.speakers_detected;

        // Reset mapper
        speakerNamesMap = {};
        
        // Show Results panel
        resultsState.classList.remove('hidden');

        // Initialize Wavesurfer Waveform
        if (wavesurfer) {
            wavesurfer.destroy();
        }

        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: '#4f46e5',
            progressColor: '#d946ef',
            cursorColor: '#f3f4f6',
            cursorWidth: 2,
            barWidth: 3,
            barGap: 2,
            barRadius: 2,
            height: 90,
            normalize: true,
            backend: 'WebAudio'
        });

        // Load files
        const objectUrl = URL.createObjectURL(audioFileObject);
        wavesurfer.load(objectUrl);

        // Wavesurfer events
        wavesurfer.on('ready', () => {
            totalTimeDisplay.textContent = formatTime(wavesurfer.getDuration());
            wavesurfer.setVolume(volumeSlider.value);
        });

        wavesurfer.on('audioprocess', (time) => {
            currentTimeDisplay.textContent = formatTime(time);
            highlightActiveChatBubble(time);
        });

        wavesurfer.on('interaction', (time) => {
            currentTimeDisplay.textContent = formatTime(wavesurfer.getCurrentTime());
            highlightActiveChatBubble(wavesurfer.getCurrentTime());
        });

        // Set up legend, timeline and chats
        renderLegend(data.conversation);
        renderTimelineRibbon(data.conversation, data.duration);
        renderChatList(data.conversation);
    }

    // Format time in mm:ss
    function formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
    }

    // Audio controls action listeners
    playBtn.addEventListener('click', () => {
        if (!wavesurfer) return;
        wavesurfer.playPause();
        const isPlaying = wavesurfer.isPlaying();
        playBtn.innerHTML = isPlaying ? '<i class="fa-solid fa-pause"></i>' : '<i class="fa-solid fa-play"></i>';
    });

    stopBtn.addEventListener('click', () => {
        if (!wavesurfer) return;
        wavesurfer.stop();
        playBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
        currentTimeDisplay.textContent = '0:00';
    });

    volumeSlider.addEventListener('input', (e) => {
        if (wavesurfer) {
            wavesurfer.setVolume(e.target.value);
        }
    });

    // Render Legend
    function renderLegend(conversation) {
        speakerLegend.innerHTML = '';
        
        // Find unique speakers
        const speakersSet = new Set();
        conversation.forEach(seg => speakersSet.add(seg.speaker));
        const uniqueSpeakers = Array.from(speakersSet).sort();

        uniqueSpeakers.forEach(spk => {
            // Get index of speaker from A, B, C, D
            const charCode = spk.split(' ')[1].charCodeAt(0) - 65;
            const colorClass = `legend-color-${charCode % 4}`;
            speakerNamesMap[spk] = spk; // Initialize name mapping

            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            legendItem.innerHTML = `
                <span class="legend-color ${colorClass}"></span>
                <span class="legend-name" contenteditable="true" data-original="${spk}">${spk}</span>
                <i class="fa-solid fa-pen" style="font-size: 9px; color: var(--text-muted);"></i>
            `;
            
            // Re-label speaker trigger listener
            const nameSpan = legendItem.querySelector('.legend-name');
            nameSpan.addEventListener('blur', (e) => {
                const newName = e.target.textContent.trim();
                const originalName = e.target.dataset.original;
                if (newName && newName !== originalName) {
                    renameSpeaker(originalName, newName);
                }
            });

            nameSpan.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    e.target.blur();
                }
            });

            speakerLegend.appendChild(legendItem);
        });
    }

    // Rename speaker throughout app
    function renameSpeaker(oldName, newName) {
        speakerNamesMap[oldName] = newName;
        
        // Update all bubble headers
        const speakerSpanElements = document.querySelectorAll(`.bubble-speaker-name[data-speaker="${oldName}"]`);
        speakerSpanElements.forEach(el => {
            el.textContent = newName;
        });

        // Update Legend names dataset targets
        const legendNames = document.querySelectorAll('.legend-name');
        legendNames.forEach(el => {
            if (el.dataset.original === oldName) {
                el.dataset.original = oldName; // keeps the track
            }
        });
    }

    // Render Timeline Ribbon
    function renderTimelineRibbon(conversation, duration) {
        timelineRibbon.innerHTML = '';
        
        conversation.forEach(seg => {
            const startPct = (seg.start / duration) * 100;
            const endPct = (seg.end / duration) * 100;
            const widthPct = endPct - startPct;

            const charCode = seg.speaker.split(' ')[1].charCodeAt(0) - 65;
            const colorClass = `legend-color-${charCode % 4}`;

            const block = document.createElement('div');
            block.className = `timeline-block ${colorClass}`;
            block.style.width = `${widthPct}%`;
            block.title = `${seg.speaker}: [${seg.start}s - ${seg.end}s]`;
            
            block.addEventListener('click', () => {
                if (wavesurfer) {
                    wavesurfer.setTime(seg.start);
                    wavesurfer.play();
                    playBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
                }
            });

            timelineRibbon.appendChild(block);
        });
    }

    // Render Chat Conversation Interface
    function renderChatList(conversation) {
        chatList.innerHTML = '';

        conversation.forEach(seg => {
            const charCode = seg.speaker.split(' ')[1].charCodeAt(0) - 65;
            const colorClass = `speaker-color-${charCode % 4}`;
            const displayName = speakerNamesMap[seg.speaker] || seg.speaker;

            const bubble = document.createElement('div');
            bubble.className = `chat-bubble ${colorClass}`;
            bubble.dataset.start = seg.start;
            bubble.dataset.end = seg.end;
            bubble.dataset.id = seg.id;
            
            bubble.innerHTML = `
                <div class="bubble-header">
                    <span class="bubble-speaker">
                        <i class="fa-solid fa-circle-user"></i>
                        <span class="bubble-speaker-name" data-speaker="${seg.speaker}">${displayName}</span>
                    </span>
                    <span class="bubble-time">[${formatTime(seg.start)} - ${formatTime(seg.end)}]</span>
                </div>
                <div class="bubble-text" contenteditable="true">${seg.text}</div>
            `;

            // Play segment on bubble click (ignoring inner text edit edits)
            bubble.addEventListener('click', (e) => {
                if (e.target.classList.contains('bubble-text') || e.target.classList.contains('bubble-speaker-name')) {
                    return; // ignore when editing text/speaker
                }
                if (wavesurfer) {
                    wavesurfer.setTime(seg.start);
                    wavesurfer.play();
                    playBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
                }
            });

            // Keep text changes inside the segment object memory
            const textDiv = bubble.querySelector('.bubble-text');
            textDiv.addEventListener('blur', (e) => {
                seg.text = e.target.textContent;
            });

            chatList.appendChild(bubble);
        });
    }

    // Highlight the chat bubble in sync with playhead time
    function highlightActiveChatBubble(time) {
        const bubbles = document.querySelectorAll('.chat-bubble');
        bubbles.forEach(bubble => {
            const start = parseFloat(bubble.dataset.start);
            const end = parseFloat(bubble.dataset.end);
            
            if (time >= start && time <= end) {
                bubble.style.filter = 'brightness(1.1)';
                bubble.style.boxShadow = '0 0 15px rgba(255, 255, 255, 0.05)';
                // Scroll active bubble into view smoothly if playing
                if (wavesurfer && wavesurfer.isPlaying()) {
                    bubble.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            } else {
                bubble.style.filter = 'none';
                bubble.style.boxShadow = 'none';
            }
        });
    }

    // Export conversation transcription as Text
    exportBtn.addEventListener('click', () => {
        if (!diarizationData) return;

        let txtContent = `Speaker Diarization & Transcription Report: ${diarizationData.filename}\n`;
        txtContent += `Duration: ${diarizationData.duration}s\n`;
        txtContent += `Speakers: ${diarizationData.speakers_detected}\n`;
        txtContent += `=========================================\n\n`;

        diarizationData.conversation.forEach(seg => {
            const displayName = speakerNamesMap[seg.speaker] || seg.speaker;
            txtContent += `[${formatTime(seg.start)} - ${formatTime(seg.end)}] ${displayName}: ${seg.text}\n`;
        });

        const blob = new Blob([txtContent], { type: 'text/plain;charset=utf-8' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        const baseName = diarizationData.filename.substring(0, diarizationData.filename.lastIndexOf('.')) || diarizationData.filename;
        link.download = `${baseName}_transcript.txt`;
        link.click();
    });

});
