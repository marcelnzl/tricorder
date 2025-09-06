document.addEventListener('DOMContentLoaded', () => {
    const transcribeBtn = document.getElementById('transcribe-btn');
    const audioFileInput = document.getElementById('audio-file-input');
    const transcriptionDisplay = document.getElementById('transcription-display');
    const summaryDisplay = document.getElementById('summary-display');
    
    // Recording elements
    const recordBtn = document.getElementById('record-btn');
    const recordingStatus = document.getElementById('recording-status');
    const audioVisualizer = document.getElementById('audio-visualizer');
    
    // Chat elements
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendChatBtn = document.getElementById('send-chat-btn');
    
    // Sidebar elements
    const meetingsList = document.getElementById('meetings-list');
    const refreshMeetingsBtn = document.getElementById('refresh-meetings-btn');
    
    // Global chat elements
    const globalChatInput = document.getElementById('global-chat-input');
    const globalChatSend = document.getElementById('global-chat-send');
    const globalChatDropdown = document.getElementById('global-chat-dropdown');
    const globalChatMessages = document.getElementById('global-chat-messages');
    const chatToggleBtn = document.getElementById('chat-toggle-btn');

    // Edit Modal elements
    const resummarizeBtn = document.getElementById('resummarize-btn');
    const editSummaryBtn = document.getElementById('edit-summary-btn');
    const editModal = document.getElementById('edit-modal');
    const closeModalBtn = editModal.querySelector('.close-btn');
    const cancelEditBtn = document.getElementById('cancel-edit-btn');
    const saveChangesBtn = document.getElementById('save-changes-btn');
    const editForm = document.getElementById('edit-form');
    
    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let audioContext = null;
    let analyser = null;
    let microphone = null;
    let animationId = null;
    
    // Chat state
    let hasNotes = false;
    let selectedMeetingId = null;
    let currentMeeting = null;
    let isGlobalChatOpen = false;

    transcribeBtn.addEventListener('click', async () => {
        const file = audioFileInput.files[0];
        if (!file) {
            transcriptionDisplay.innerHTML = '<p style="color: #ff453a;">Please select an audio file first.</p>';
            return;
        }

        // Prepare the form data to send the file
        const formData = new FormData();
        formData.append('file', file);

        // Update UI to show processing state
        transcriptionDisplay.innerHTML = '<p>Uploading and transcribing, please wait...</p>';
        summaryDisplay.innerHTML = '<p>Awaiting transcription to generate summary...</p>';

        try {
            // Call the backend API
            const response = await fetch('/transcribe', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'An unknown error occurred.');
            }

            const result = await response.json();
            
            // Display the transcription
            transcriptionDisplay.textContent = result.transcription;

            // Call summarization endpoint
            await summarizeTranscription(result.transcription);

        } catch (error) {
            console.error('Error during transcription:', error);
            transcriptionDisplay.innerHTML = `<p style="color: #ff453a;">Error: ${error.message}</p>`;
        }
    });

    // Initialize audio visualizer
    function initializeVisualizer() {
        audioVisualizer.innerHTML = '';
        for (let i = 0; i < 20; i++) {
            const bar = document.createElement('div');
            bar.className = 'visualizer-bar';
            bar.style.height = '12px';
            audioVisualizer.appendChild(bar);
        }
    }

    // Update audio visualizer based on audio input
    function updateVisualizer() {
        if (!analyser || !isRecording) return;

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyser.getByteFrequencyData(dataArray);

        const bars = audioVisualizer.querySelectorAll('.visualizer-bar');

        // Use a more focused frequency range for better visualization
        const startFreq = Math.floor(bufferLength * 0.1); // Skip very low frequencies
        const endFreq = Math.floor(bufferLength * 0.8);   // Skip very high frequencies
        const freqRange = endFreq - startFreq;

        bars.forEach((bar, index) => {
            // Map bar index to frequency range
            const freqIndex = startFreq + Math.floor((index / bars.length) * freqRange);
            const value = dataArray[Math.min(freqIndex, bufferLength - 1)];

            // Double the height range for more dramatic effect
            const height = Math.max(12, (value / 255) * 120);
            const opacity = Math.max(0.4, Math.min(1.0, value / 200));

            bar.style.height = `${height}px`;
            bar.style.opacity = opacity;
            bar.style.transform = `scaleY(${0.8 + (value / 255) * 0.4})`;
        });

        if (isRecording) {
            animationId = requestAnimationFrame(updateVisualizer);
        }
    }

    // Handle recording button click
    recordBtn.addEventListener('click', async () => {
        if (!isRecording) {
            try {
		// List of potential mimeTypes in order of preference
            const mimeTypes = [
                'audio/webm;codecs=opus', // Great for quality and size
                'audio/mp4',             // Best for Apple devices
                'audio/webm',            // Fallback for webm
            ];

            // Find the first supported mimeType
            const supportedMimeType = mimeTypes.find(type => MediaRecorder.isTypeSupported(type));

            if (!supportedMimeType) {
                alert('Audio recording is not supported on this browser.');
                return;
            }
                // Request microphone access
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        sampleRate: 44100
                    } 
                });

                // Set up audio context for visualization
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                microphone = audioContext.createMediaStreamSource(stream);
                microphone.connect(analyser);
                analyser.fftSize = 256;

                // Set up MediaRecorder
                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: supportedMimeType
                });

                audioChunks = [];

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = async () => {
                    // Create audio blob from recorded chunks
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    
                    // Convert to a format suitable for transcription
                    await transcribeRecordedAudio(audioBlob);
                    
                    // Clean up
                    stream.getTracks().forEach(track => track.stop());
                    if (audioContext) {
                        audioContext.close();
                    }
                };

                // Start recording
                mediaRecorder.start();
                isRecording = true;

                // Update UI
                recordBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12" rx="2" ry="2"></rect></svg>';
                recordBtn.classList.add('recording');
                recordingStatus.textContent = '🔴 Recording in progress...';
                
                // Initialize and show visualizer
                initializeVisualizer();
                audioVisualizer.classList.add('active');
                updateVisualizer();

            } catch (error) {
                console.error('Error accessing microphone:', error);
                recordingStatus.textContent = 'Error: Could not access microphone';
                recordingStatus.style.color = '#ff453a';
            }
        } else {
            // Stop recording
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
            
            isRecording = false;
            
            // Update UI
            recordBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>';
            recordBtn.classList.remove('recording');
            recordingStatus.textContent = 'Processing recording...';
            audioVisualizer.classList.remove('active');
            
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
        }
    });

    // Transcribe recorded audio
    async function transcribeRecordedAudio(audioBlob) {
        try {
            // Convert webm to a file-like object
            const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' });
            
            // Prepare form data
            const formData = new FormData();
            formData.append('file', audioFile);

            // Update UI
            transcriptionDisplay.innerHTML = '<p>Transcribing your recording, please wait...</p>';
            summaryDisplay.innerHTML = '<p>Awaiting transcription to generate summary...</p>';
            recordingStatus.textContent = 'Transcribing...';

            // Call the transcription API
            const response = await fetch('/transcribe', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Transcription failed');
            }

            const result = await response.json();
            
            // Display results
            transcriptionDisplay.textContent = result.transcription;
            recordingStatus.textContent = 'Recording complete!';
            recordingStatus.style.color = '#30d158';

            // Call summarization endpoint
            await summarizeTranscription(result.transcription);

            // Reset status after a few seconds
            setTimeout(() => {
                recordingStatus.textContent = '';
                recordingStatus.style.color = '#ff453a';
            }, 3000);

        } catch (error) {
            console.error('Error transcribing recorded audio:', error);
            transcriptionDisplay.innerHTML = `<p style="color: #ff453a;">Error: ${error.message}</p>`;
            recordingStatus.textContent = 'Transcription failed';
            recordingStatus.style.color = '#ff453a';
        }
    }

    // Summarize transcribed text using Gemini AI
    async function summarizeTranscription(transcriptionText) {
        if (!transcriptionText || transcriptionText.trim().length === 0) {
            summaryDisplay.innerHTML = '<p style="color: #ff453a;">No transcription text to summarize.</p>';
            return;
        }

        try {
            // Update UI to show processing
            summaryDisplay.innerHTML = '<p>Generating AI summary with Gemini 2.0 Flash...</p>';

            // Call the summarization API
            const response = await fetch('/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: transcriptionText
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Summarization failed');
            }

            const result = await response.json();
            
            // Format and display the summary
            let summaryHTML = '<div class="summary-content">';

            // Check if result.data is a valid object and not empty
            if (result.data && typeof result.data === 'object' && Object.keys(result.data).length > 0) {
                for (const [key, value] of Object.entries(result.data)) {
                    const title = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                    summaryHTML += `<h3>${title}</h3>`;

                    if (Array.isArray(value)) {
                        summaryHTML += '<ul>';
                        value.forEach(item => {
                            summaryHTML += `<li>${item}</li>`;
                        });
                        summaryHTML += '</ul>';
                    } else {
                        // If the value is an object, stringify it for display
                        summaryHTML += `<p>${typeof value === 'object' ? JSON.stringify(value, null, 2) : value}</p>`;
                    }
                }
            } else {
                // If data is not in expected format, display raw JSON
                summaryHTML += `<h3>Raw Summary Data</h3><pre>${JSON.stringify(result, null, 2)}</pre>`;
            }
            
            summaryHTML += `</div>`;
            
            summaryDisplay.innerHTML = summaryHTML;

            // Enable chat interface after first successful transcription
            if (!hasNotes) {
                enableChatInterface();
                hasNotes = true;
            }

            // Refresh meetings list to show the new meeting
            await loadMeetings();

            // Automatically select the new meeting to make it editable
            if (result.note_id) {
                await selectMeeting(result.note_id);
            }

        } catch (error) {
            console.error('Error during summarization:', error);
            summaryDisplay.innerHTML = `<p style="color: #ff453a;">Summarization Error: ${error.message}</p>`;
        }
    }

    // Utility function to convert blob to different format if needed
    async function convertAudioFormat(blob, targetFormat = 'wav') {
        return new Promise((resolve, reject) => {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const fileReader = new FileReader();
            
            fileReader.onload = async (e) => {
                try {
                    const arrayBuffer = e.target.result;
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    
                    // For now, just return the original blob
                    // In a full implementation, you'd convert the format here
                    resolve(blob);
                } catch (error) {
                    reject(error);
                }
            };
            
            fileReader.onerror = reject;
            fileReader.readAsArrayBuffer(blob);
        });
    }

    // Chat functionality
    function enableChatInterface() {
        chatInput.disabled = false;
        sendChatBtn.disabled = false;
        chatInput.placeholder = "Ask a question about your notes...";
        
        // Clear welcome message
        const welcomeMessage = chatMessages.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.style.display = 'none';
        }
        
        // Add initial AI message
        addChatMessage('ai', "Great! I've indexed your notes. You can now ask me questions about your transcriptions. What would you like to know?");
    }

    function addChatMessage(type, content, sources = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${type}-message`;
        
        if (type === 'ai' && sources.length > 0) {
            let sourcesHTML = '<div class="sources"><strong>Sources:</strong>';
            sources.forEach(source => {
                const timestamp = new Date(source.timestamp).toLocaleString();
                const sourceType = source.type || 'Note'; // Default to 'Note' if type isn't specified
                sourcesHTML += `
                    <div class="source-item" data-meeting-id="${source.note_id}">
                        <a href="#" class="source-link" data-meeting-id="${source.note_id}">
                            📄 ${sourceType} from ${timestamp} (Score: ${source.relevance_score})
                        </a>
                        <em>${source.preview}</em>
                    </div>`;
            });
            sourcesHTML += '</div>';
            messageDiv.innerHTML = content + sourcesHTML;
        } else {
            messageDiv.textContent = content;
        }
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Add event listeners to source links
        addSourceLinkListeners(messageDiv);
    }

    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 6px; vertical-align: middle;"><rect x="3" y="11" width="18" height="10" rx="2" ry="2"></rect><circle cx="12" cy="16" r="1"></circle><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>AI is thinking<div class="typing-dots"><span></span><span></span><span></span></div>';
        typingDiv.id = 'typing-indicator';
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    async function sendChatMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        // Add user message to chat
        addChatMessage('user', message);
        
        // Clear input and disable while processing
        chatInput.value = '';
        chatInput.disabled = true;
        sendChatBtn.disabled = true;
        
        // Show typing indicator
        showTypingIndicator();

        try {
            // Call chat API
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    meeting_id: selectedMeetingId
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Chat request failed');
            }

            const result = await response.json();
            
            // Hide typing indicator
            hideTypingIndicator();
            
            // Add AI response to chat
            addChatMessage('ai', result.response, result.sources);

        } catch (error) {
            console.error('Error during chat:', error);
            hideTypingIndicator();
            addChatMessage('ai', `Sorry, I encountered an error: ${error.message}`);
        } finally {
            // Re-enable input
            chatInput.disabled = false;
            sendChatBtn.disabled = false;
            chatInput.focus();
        }
    }

    // Chat event listeners
    sendChatBtn.addEventListener('click', sendChatMessage);
    
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !chatInput.disabled) {
            sendChatMessage();
        }
    });

    // Focus chat input when user starts typing
    chatInput.addEventListener('focus', () => {
        if (!hasNotes) {
            chatInput.blur();
        }
    });

    // Source Link Handling
    function addSourceLinkListeners(messageElement) {
        const sourceLinks = messageElement.querySelectorAll('.source-link');
        sourceLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const meetingId = link.dataset.meetingId;
                if (meetingId) {
                    handleSourceClick(meetingId);
                }
            });
        });
    }

    async function handleSourceClick(meetingId) {
        try {
            // Close global chat dropdown if it's open
            if (isGlobalChatOpen) {
                toggleGlobalChat();
            }

            // Select the meeting in the sidebar
            await selectMeeting(meetingId);
            
            // Optional: Add visual feedback
            const selectedItem = meetingsList.querySelector(`[data-meeting-id="${meetingId}"]`);
            if (selectedItem) {
                // Add a temporary highlight effect
                selectedItem.style.animation = 'highlight 1s ease-out';
                setTimeout(() => {
                    selectedItem.style.animation = '';
                }, 1000);
            }

        } catch (error) {
            console.error('Error navigating to source meeting:', error);
            // Show a user-friendly message
            if (isGlobalChatOpen) {
                addGlobalChatMessage('ai', 'Sorry, I had trouble loading that meeting. Please try clicking on it in the sidebar.');
            } else {
                addChatMessage('ai', 'Sorry, I had trouble loading that meeting. Please try clicking on it in the sidebar.');
            }
        }
    }

    // Global Chat Functions
    function toggleGlobalChat() {
        isGlobalChatOpen = !isGlobalChatOpen;
        if (isGlobalChatOpen) {
            globalChatDropdown.classList.add('open');
        } else {
            globalChatDropdown.classList.remove('open');
        }
    }

    function addGlobalChatMessage(type, content, sources = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `global-chat-message global-${type}-message`;
        
        if (type === 'ai' && sources.length > 0) {
            let sourcesHTML = '<div class="sources"><strong>Sources:</strong>';
            sources.forEach(source => {
                const timestamp = new Date(source.timestamp).toLocaleString();
                const sourceType = source.type || 'Note';
                sourcesHTML += `
                    <div class="source-item" data-meeting-id="${source.note_id}">
                        <a href="#" class="source-link" data-meeting-id="${source.note_id}">
                            📄 ${sourceType} from ${timestamp} (Score: ${source.relevance_score})
                        </a>
                        <em>${source.preview}</em>
                    </div>`;
            });
            sourcesHTML += '</div>';
            messageDiv.innerHTML = content + sourcesHTML;
        } else {
            messageDiv.textContent = content;
        }
        
        globalChatMessages.appendChild(messageDiv);
        globalChatMessages.scrollTop = globalChatMessages.scrollHeight;
        
        // Add event listeners to source links
        addSourceLinkListeners(messageDiv);
    }

    function showGlobalTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'global-typing-indicator';
        typingDiv.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 6px; vertical-align: middle;"><rect x="3" y="11" width="18" height="10" rx="2" ry="2"></rect><circle cx="12" cy="16" r="1"></circle><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>AI is thinking<div class="global-typing-dots"><span></span><span></span><span></span></div>';
        typingDiv.id = 'global-typing-indicator';
        globalChatMessages.appendChild(typingDiv);
        globalChatMessages.scrollTop = globalChatMessages.scrollHeight;
    }

    function hideGlobalTypingIndicator() {
        const typingIndicator = document.getElementById('global-typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    async function sendGlobalChatMessage() {
        const message = globalChatInput.value.trim();
        if (!message) return;

        // Open dropdown if not already open
        if (!isGlobalChatOpen) {
            toggleGlobalChat();
        }

        // Hide welcome message on first use
        const welcomeMessage = globalChatMessages.querySelector('.global-welcome-message');
        if (welcomeMessage) {
            welcomeMessage.style.display = 'none';
        }

        // Add user message to global chat
        addGlobalChatMessage('user', message);
        
        // Clear input and disable while processing
        globalChatInput.value = '';
        globalChatInput.disabled = true;
        globalChatSend.disabled = true;
        
        // Show typing indicator
        showGlobalTypingIndicator();

        try {
            // Call chat API (without meeting_id to search all notes)
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message
                    // No meeting_id = search all notes
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Chat request failed');
            }

            const result = await response.json();
            
            // Hide typing indicator
            hideGlobalTypingIndicator();
            
            // Add AI response to global chat
            addGlobalChatMessage('ai', result.response, result.sources);

        } catch (error) {
            console.error('Error during global chat:', error);
            hideGlobalTypingIndicator();
            addGlobalChatMessage('ai', `Sorry, I encountered an error: ${error.message}`);
        } finally {
            // Re-enable input
            globalChatInput.disabled = false;
            globalChatSend.disabled = false;
            globalChatInput.focus();
        }
    }

    // Global chat event listeners
    globalChatSend.addEventListener('click', sendGlobalChatMessage);
    
    globalChatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !globalChatInput.disabled) {
            sendGlobalChatMessage();
        }
    });

    // Close global chat when clicking outside
    document.addEventListener('click', (e) => {
        const isClickInsideGlobalChat = globalChatDropdown.contains(e.target) ||
                                       e.target.closest('.global-chat-header');

        if (!isClickInsideGlobalChat && isGlobalChatOpen) {
            toggleGlobalChat();
        }
    });

    // Chat toggle event listeners
    if (chatToggleBtn) {
        chatToggleBtn.addEventListener('click', toggleGlobalChat);
    }

    // Meeting management functions
    async function loadMeetings() {
        try {
            const response = await fetch('/meetings');
            if (!response.ok) {
                throw new Error('Failed to load meetings');
            }

            const data = await response.json();
            displayMeetings(data.meetings);

            // If meetings exist on load, enable the chat interface
            if (data.meetings && data.meetings.length > 0 && !hasNotes) {
                enableChatInterface();
                hasNotes = true;
            }
        } catch (error) {
            console.error('Error loading meetings:', error);
            meetingsList.innerHTML = '<div class="no-meetings"><p>Error loading meetings</p></div>';
        }
    }

    function displayMeetings(meetings) {
        if (!meetings || meetings.length === 0) {
            meetingsList.innerHTML = '<div class="no-meetings"><p>No meetings yet. Start by recording or uploading audio to create your first meeting summary.</p></div>';
            return;
        }

        let html = '';
        meetings.forEach(meeting => {
            const date = new Date(meeting.created_at);
            const formattedDate = date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });

            html += `
                <div class="meeting-item" data-meeting-id="${meeting.meeting_id}">
                    <div class="meeting-item-content">
                        <div class="meeting-title">${meeting.title}</div>
                        <div class="meeting-summary">${meeting.summary}</div>
                        <div class="meeting-date">${formattedDate}</div>
                    </div>
                    <button class="delete-meeting-btn" data-meeting-id="${meeting.meeting_id}" title="Delete meeting">
                        <span>&times;</span>
                    </button>
                </div>
            `;
        });

        meetingsList.innerHTML = html;
    }

    async function selectMeeting(meetingId) {
        try {
            // Update UI to show selection
            const allItems = meetingsList.querySelectorAll('.meeting-item');
            allItems.forEach(item => item.classList.remove('selected'));
            
            const selectedItem = meetingsList.querySelector(`[data-meeting-id="${meetingId}"]`);
            if (selectedItem) {
                selectedItem.classList.add('selected');
            }

            // Load meeting details
            const response = await fetch(`/meetings/${meetingId}`);
            if (!response.ok) {
                throw new Error('Failed to load meeting details');
            }

            const meeting = await response.json();
            selectedMeetingId = meetingId;
            currentMeeting = meeting;

            // Update main content area
            loadMeetingContent(meeting);

            // Clear chat and show meeting-specific welcome
            clearChatMessages();
            const welcomeMsg = selectedMeetingId 
                ? `Now chatting about: "${meeting.title}". Ask me anything about this specific meeting!`
                : "You can now ask questions about all your meetings.";
            addChatMessage('ai', welcomeMsg);

        } catch (error) {
            console.error('Error selecting meeting:', error);
            addChatMessage('ai', 'Sorry, I had trouble loading that meeting.');
        }
    }

    function loadMeetingContent(meeting) {
        // Update transcription display
        transcriptionDisplay.textContent = meeting.transcription;

        // Update summary display with structured format
        let summaryHTML = '<div class="summary-content">';

        // Handle both old and new data structures
        const data = meeting.data || {
            summary: meeting.summary,
            key_points: meeting.key_points,
            action_items: meeting.action_items,
            categories: meeting.categories
        };

        // Check if data is a valid object and not empty
        if (data && typeof data === 'object' && Object.keys(data).length > 0) {
            for (const [key, value] of Object.entries(data)) {
                if (value) {
                    const title = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                    summaryHTML += `<h3>${title}</h3>`;

                    if (Array.isArray(value)) {
                        summaryHTML += '<ul>';
                        value.forEach(item => {
                            summaryHTML += `<li>${item}</li>`;
                        });
                        summaryHTML += '</ul>';
                    } else {
                        // If the value is an object, stringify it for display
                        summaryHTML += `<p>${typeof value === 'object' ? JSON.stringify(value, null, 2) : value}</p>`;
                    }
                }
            }
        } else {
            // If data is not in expected format, display raw JSON
            summaryHTML += `<h3>Raw Meeting Data</h3><pre>${JSON.stringify(meeting, null, 2)}</pre>`;
        }
        
        summaryHTML += `</div>`;
        
        summaryDisplay.innerHTML = summaryHTML;

        // Show the edit and resummarize buttons
        editSummaryBtn.style.display = 'inline-block';
        resummarizeBtn.style.display = 'inline-block';
    }

    function clearChatMessages() {
        const messages = chatMessages.querySelectorAll('.chat-message, .typing-indicator');
        messages.forEach(msg => msg.remove());
    }

    function deselectMeeting() {
        // Clear selection
        const allItems = meetingsList.querySelectorAll('.meeting-item');
        allItems.forEach(item => item.classList.remove('selected'));
        
        selectedMeetingId = null;
        currentMeeting = null;
        
        // Clear main content
        transcriptionDisplay.innerHTML = '<p>Select a meeting from the sidebar to view its content</p>';
        summaryDisplay.innerHTML = '<p>Select a meeting from the sidebar to view its summary</p>';
        
        // Hide the edit and resummarize buttons
        editSummaryBtn.style.display = 'none';
        resummarizeBtn.style.display = 'none';

        // Update chat welcome message
        clearChatMessages();
        addChatMessage('ai', "You can now ask questions about all your meetings. Select a specific meeting from the sidebar to focus the conversation.");
    }

    // Event listeners for sidebar
    refreshMeetingsBtn.addEventListener('click', loadMeetings);

    // Use a single event listener for the entire meetings list (event delegation)
    meetingsList.addEventListener('click', (e) => {
        const meetingItem = e.target.closest('.meeting-item');
        if (!meetingItem) return;

        const meetingId = meetingItem.dataset.meetingId;

        if (e.target.closest('.delete-meeting-btn')) {
            // Handle delete button click
            handleDeleteClick(meetingId, e.target.closest('.delete-meeting-btn'));
        } else {
            // Handle meeting selection
            selectMeeting(meetingId);
        }
    });

    // Delete Meeting Functionality
    function handleDeleteClick(meetingId, deleteButton) {
        // Prevent the meeting selection from firing
        event.stopPropagation();

        if (confirm('Are you sure you want to permanently delete this meeting and its transcription?')) {
            deleteMeeting(meetingId);
        }
    }

    async function deleteMeeting(meetingId) {
        try {
            const response = await fetch(`/meetings/${meetingId}`, {
                method: 'DELETE',
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to delete meeting');
            }

            // If the deleted meeting was the selected one, clear the view
            if (selectedMeetingId === meetingId) {
                deselectMeeting();
            }

            // Refresh the list of meetings
            await loadMeetings();

        } catch (error) {
            console.error('Error deleting meeting:', error);
            alert(`Error: ${error.message}`);
        }
    }

    // Edit Modal Functions
    function openEditModal() {
        if (!currentMeeting) return;

        // Populate the form with current meeting data
        document.getElementById('edit-title').value = currentMeeting.title;
        const dataToEdit = currentMeeting.data || {
            summary: currentMeeting.summary,
            key_points: currentMeeting.key_points,
            action_items: currentMeeting.action_items,
            categories: currentMeeting.categories
        };
        document.getElementById('edit-data').value = JSON.stringify(dataToEdit, null, 2);

        editModal.style.display = 'block';
    }

    function closeEditModal() {
        editModal.style.display = 'none';
    }

    async function saveMeetingChanges() {
        if (!currentMeeting) return;

        // Get data from form
        const title = document.getElementById('edit-title').value;
        let data;
        try {
            data = JSON.parse(document.getElementById('edit-data').value);
        } catch (e) {
            alert("Invalid JSON format in data field.");
            return;
        }

        const updatedData = {
            title,
            data
        };

        try {
            const response = await fetch(`/meetings/${currentMeeting.meeting_id}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(updatedData),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to save changes');
            }

            // Close modal and refresh content
            closeEditModal();
            await loadMeetings(); // Refresh sidebar in case title changed
            await selectMeeting(currentMeeting.meeting_id); // Reload the meeting view

        } catch (error) {
            console.error('Error saving meeting changes:', error);
            alert(`Error: ${error.message}`);
        }
    }

    // Edit modal event listeners
    resummarizeBtn.addEventListener('click', handleResummarize);
    editSummaryBtn.addEventListener('click', openEditModal);
    closeModalBtn.addEventListener('click', closeEditModal);
    cancelEditBtn.addEventListener('click', closeEditModal);
    saveChangesBtn.addEventListener('click', saveMeetingChanges);

    async function handleResummarize() {
        if (!currentMeeting) {
            alert("Please select a meeting to re-summarize.");
            return;
        }

        if (!confirm("Are you sure you want to reprocess the summary? This will replace the current summary with a new one.")) {
            return;
        }

        try {
            summaryDisplay.innerHTML = '<p>Reprocessing summary, please wait...</p>';

            const response = await fetch(`/meetings/${currentMeeting.meeting_id}/resummarize`, {
                method: 'POST',
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to reprocess summary');
            }

            const result = await response.json();

            // Update the current meeting data and reload the view
            await selectMeeting(result.meeting_id);
            // Also refresh the meetings list in case the title changed
            await loadMeetings();

        } catch (error) {
            console.error('Error during re-summarization:', error);
            summaryDisplay.innerHTML = `<p style="color: #ff453a;">Error: ${error.message}</p>`;
        }
    }

    // Close modal if clicking outside of it
    window.addEventListener('click', (event) => {
        if (event.target == editModal) {
            closeEditModal();
        }
    });

    // Load meetings on page load
    loadMeetings();

    // Add keyboard shortcut to deselect meeting (Escape key)
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (editModal.style.display === 'block') {
                closeEditModal();
            } else if (selectedMeetingId) {
                deselectMeeting();
            }
        }
    });

    // Mobile menu toggle functionality
    const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
    const sidebar = document.querySelector('.sidebar');
    const sidebarOverlay = document.querySelector('.sidebar-overlay');

    function toggleMobileMenu() {
        sidebar.classList.toggle('active');
        sidebarOverlay.classList.toggle('active');
    }

    function closeMobileMenu() {
        sidebar.classList.remove('active');
        sidebarOverlay.classList.remove('active');
    }

    if (mobileMenuToggle) {
        mobileMenuToggle.addEventListener('click', toggleMobileMenu);
    }

    if (sidebarOverlay) {
        sidebarOverlay.addEventListener('click', closeMobileMenu);
    }

    // Close mobile menu when clicking on a meeting item
    meetingsList.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && e.target.closest('.meeting-item')) {
            closeMobileMenu();
        }
    });
});
