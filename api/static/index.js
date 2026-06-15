document.addEventListener('DOMContentLoaded', () => {
    const ticketTextarea = document.getElementById('ticket-text');
    const classifyBtn = document.getElementById('classify-btn');
    const resultCard = document.getElementById('result-card');
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsData = document.getElementById('results-data');
    const predictedCategory = document.getElementById('predicted-category');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidencePercentage = document.getElementById('confidence-percentage');
    const predictedPriority = document.getElementById('predicted-priority');
    const priorityConfidenceBar = document.getElementById('priority-confidence-bar');
    const priorityConfidencePercentage = document.getElementById('priority-confidence-percentage');
    const predictedSentiment = document.getElementById('predicted-sentiment');
    const sentimentConfidenceBar = document.getElementById('sentiment-confidence-bar');
    const sentimentConfidencePercentage = document.getElementById('sentiment-confidence-percentage');
    const reasoningTermsContainer = document.getElementById('reasoning-terms');
    const suggestedAction = document.getElementById('suggested-action');
    const historyRows = document.getElementById('history-rows');
    const badgeChips = document.querySelectorAll('.badge-chip');

    // 1. Quick Chips Click Event
    badgeChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const presetText = chip.getAttribute('data-text');
            ticketTextarea.value = presetText;
            ticketTextarea.focus();
        });
    });

    // 2. Classify Action
    classifyBtn.addEventListener('click', async () => {
        const text = ticketTextarea.value.trim();

        if (!text) {
            alert('Please enter some ticket text or choose a ticket template shortcut.');
            return;
        }

        // Set Loading State
        classifyBtn.disabled = true;
        classifyBtn.textContent = 'PROCESSING...';
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to process ticket.');
            }

            const data = await response.json();
            displayResult(data.category, data.confidence, data.priority, data.priority_confidence, data.sentiment, data.sentiment_confidence, data.reasoning, data.suggested_action);
            addHistoryEntry(text, data.category, data.confidence, data.priority, data.priority_confidence, data.sentiment, data.sentiment_confidence);

        } catch (error) {
            console.error('Error during routing:', error);
            alert(`Error: ${error.message}`);
        } finally {
            // Restore Button State
            classifyBtn.disabled = false;
            classifyBtn.textContent = 'PROCESS & ROUTE';
        }
    });

    // Display prediction result in the card
    function displayResult(category, confidence, priority, priorityConfidence, sentiment, sentimentConfidence, reasoning, routingAction) {
        // Toggle view containers
        resultsPlaceholder.classList.add('hidden');
        resultsData.classList.remove('hidden');
        resultCard.classList.remove('empty-state');

        // Populate values
        predictedCategory.textContent = category;
        
        const percentage = Math.round(confidence * 100);
        confidenceBar.style.width = `${percentage}%`;
        confidencePercentage.textContent = `${percentage}%`;

        // Populate priority values
        predictedPriority.textContent = priority;
        predictedPriority.className = 'priority-badge ' + priority.toLowerCase();

        const priorityPercentage = Math.round(priorityConfidence * 100);
        priorityConfidenceBar.style.width = `${priorityPercentage}%`;
        priorityConfidencePercentage.textContent = `${priorityPercentage}%`;

        // Populate sentiment values
        predictedSentiment.textContent = sentiment;
        predictedSentiment.className = 'sentiment-badge ' + sentiment.toLowerCase();

        const sentimentPercentage = Math.round(sentimentConfidence * 100);
        sentimentConfidenceBar.style.width = `${sentimentPercentage}%`;
        sentimentConfidencePercentage.textContent = `${sentimentPercentage}%`;

        // Populate reasoning chips
        reasoningTermsContainer.innerHTML = '';
        if (reasoning && reasoning.length > 0) {
            reasoning.forEach(term => {
                const chip = document.createElement('span');
                chip.className = 'reasoning-chip';
                chip.textContent = term;
                reasoningTermsContainer.appendChild(chip);
            });
        } else {
            reasoningTermsContainer.innerHTML = '<span class="caption">No key terms detected</span>';
        }

        // Populate suggested action
        suggestedAction.textContent = routingAction;
    }

    // Add entry to history log
    function addHistoryEntry(text, category, confidence, priority, priorityConfidence, sentiment, sentimentConfidence) {
        // Remove empty history row if present
        const emptyRow = historyRows.querySelector('.empty-history-row');
        if (emptyRow) {
            emptyRow.remove();
        }

        // Create new row
        const row = document.createElement('tr');
        
        // Snip the text snippet
        const snippet = text.length > 50 ? text.substring(0, 50) + '...' : text;
        const percentage = Math.round(confidence * 100);

        row.innerHTML = `
            <td>${escapeHTML(snippet)}</td>
            <td>
                <strong style="text-transform: uppercase; color: var(--primary);">${category}</strong>
                <span class="caption" style="font-size: 11px; font-weight: 600; text-transform: uppercase; margin-left: 2px;">(${priority})</span>
                <span class="caption" style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: var(--body); margin-left: 2px;">[${sentiment}]</span>
            </td>
            <td><strong>${percentage}%</strong></td>
        `;

        // Insert at the beginning of the table body
        historyRows.insertBefore(row, historyRows.firstChild);

        // Limit history to 5 rows
        if (historyRows.children.length > 5) {
            historyRows.removeChild(historyRows.lastChild);
        }
    }

    // Simple HTML escaping helper to prevent XSS
    function escapeHTML(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
});
