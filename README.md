## API Endpoints Used by the AI Tutor Frontend

The AI Tutor frontend communicates with the backend through REST API endpoints hosted on Railway. These endpoints support interactive tutoring, handwritten answer preparation workflows, voice-based conversations, student access checks, preparation reports, response analysis, and usage analytics.

**Production API Base URL**

```text
https://usefulapis-production.up.railway.app
```

The frontend sends HTTP requests to the endpoints listed below. The methods and paths reflect the frontend request sites identified during code inspection.

### 1. AI Interactive Learning

These endpoints support the text-based, question-specific tutoring workflow.

| Method | Endpoint | Frontend Component |
|---|---|---|
| `POST` | `/send_message_anz_way_model_evaluation` | `Ai_Learning.js` |
| `POST` | `/chat_anz_way_model_evaluation` | `Ai_Learning.js` |

### 2. AI Tutor Training and Interactive Chat

These endpoints support the training and conversational tutoring functionality.

| Method | Endpoint | Frontend Component |
|---|---|---|
| `POST` | `/train-on-images-anz-way-new` | `ChatbotTrainerUI_sociology.js` |
| `POST` | `/chat_interactive_tutor` | `ChatbotTrainerUI_sociology.js` |

### 3. Voice-Based AI Learning

These endpoints support audio-enabled tutoring interactions, including sending audio messages, processing voice-based conversations, and retrieving generated audio.

| Method | Endpoint | Frontend Component |
|---|---|---|
| `POST` | `/chat_anz_way_model_evaluation_audio` | `AiAudioLearning.js` |
| `GET` | `/get-audio/{session_id}` | `AiAudioLearning.js` |
| `POST` | `/send_audio_message` | `AiAudioLearning.js` |

The `{session_id}` placeholder represents the session identifier supplied by the frontend when requesting audio.

### 4. Student Access and Examination Questions

These endpoints support student access checks and retrieval of questions for student reporting workflows.

| Method | Endpoint | Frontend Component |
|---|---|---|
| `GET` | `/check-user-access?username={username}` | `StudentDashboard.js` |
| `GET` | `/questions_a_level` | `StudentReport.js` |

The `username` query parameter identifies the user for the access-check request.

### 5. Student Reports and Response Analysis

These endpoints support the retrieval and generation of student reporting information.

| Method | Endpoint | Frontend Component |
|---|---|---|
| `GET` | `/response_analyzer_anzway?username={username}&from_date={from_date}&to_date={to_date}` | `ResponseAnalyzer.js` |
| `POST` | `/student_report` | `StudentReport.js` |

The Response Analyzer endpoint accepts the following query parameters:

- `username` — identifies the student whose interactions are being analysed.
- `from_date` — specifies the beginning of the reporting period.
- `to_date` — specifies the end of the reporting period.

### 6. Usage Analytics and Reporting

These endpoints support usage-related reporting and the retrieval of user usage information.

| Method | Endpoint | Frontend Component |
|---|---|---|
| `POST` | `/student_total_usage` | `StudentUsageReport.js` |
| `GET` | `/users_total_usage?min_usage={min_usage}&max_usage={max_usage}` | `UserUsageDashboard.js` |

The `/users_total_usage` endpoint accepts two query parameters:

- `min_usage` — specifies the minimum usage threshold.
- `max_usage` — specifies the maximum usage threshold.

### API Integration Summary

The frontend integrates with the backend across six functional areas:

- **Interactive learning:** Text-based AI tutoring and examination preparation.
- **Training and tutoring:** Training-related operations and interactive chat.
- **Voice-based learning:** Audio messaging, voice tutoring, and audio retrieval.
- **Student access and questions:** User access checks and examination-question retrieval.
- **Student reporting:** Historical response analysis and student reports.
- **Usage analytics:** Student usage reporting and usage-range filtering.

These endpoints form the API integration surface used by the documented frontend components. Request payloads, response schemas, authentication requirements, and detailed validation rules should be documented alongside the corresponding backend route implementations.

