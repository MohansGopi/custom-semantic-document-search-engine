# TF-IDF Retrieval-Augmented Generation (RAG) System

This is a custom, file-based system built with **FastAPI** that implements a basic **Retrieval-Augmented Generation (RAG)** pipeline. It uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert documents and queries into numerical vectors, and **Cosine Similarity** for efficient information retrieval.

---

## ⚙️ Setup Instructions

### 1. Prerequisites

* Python 3.8 or higher must be installed.
* Ensure `virtualenv` is installed (`pip install virtualenv`) to create isolated environments.

### 2. Create Environment and Install Dependencies

1. Navigate to the project root directory.
2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

* On Windows:

```bash
venv\Scripts\activate
```

* On macOS/Linux:

```bash
source venv/bin/activate
```

4. Install the required Python packages:

```bash
pip install fastapi uvicorn python-dotenv numpy
```

### 3. Configuration and Data Setup

The system relies on two folders defined in your `.env` file:

| Variable         | Default Value | Description                                                              |
| ---------------- | ------------- | ------------------------------------------------------------------------ |
| `PATH_DOCUMENTS` | `./documents` | Folder where all your input `.txt` files must be placed.                 |
| `MODEL_FOLDER`   | `./model`     | Directory where processed vocabulary, IDF, and TF-IDF vectors are saved. |

#### Setup Steps

1. **Create Directories**:

```bash
mkdir documents model
```

2. **Add Documents**:
   Place all your plain text files (`.txt`) into the `./documents` folder.

> **Note:** The system will use the cleaned filename as the document topic.

### 4. Running the Server

Start the FastAPI application using `uvicorn`:

```bash
uvicorn server:app --reload
```

The application will be available at:

```
http://127.0.0.1:8000
```

---

## 🚀 Usage Guide (API Calls)

### Step 1: Index and Process Documents (Mandatory)

You must run this endpoint **once** before performing any searches, and re-run it whenever you change the documents in the `./documents` folder.
This builds the vector database.

* **Endpoint:** `GET /index`
* **Action:** Triggers full TF-IDF vectorization workflow.

**Sample API Call:**

```bash
curl http://127.0.0.1:8000/index
```

**Responses:**

| Condition                    | Response Body                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| Success (Documents Indexed)  | `{"Status": "Ok"}`                                                                 |
| Failure (No Documents Found) | `{"Status": "Not Okay", "Status Code": 400, "Message": "Please upload any files"}` |

---

### Step 2: Search Documents

After successful indexing, use the `/search` endpoint with your query string.

* **Endpoint:** `GET /search`
* **Query Parameter:** `q` (The text query string)

**Sample API Call:**

```bash
curl "http://127.0.0.1:8000/search?q=how ai helps in agriculture"
```

**Success Response (200 OK):**

Returns the top 3 most relevant documents with their Cosine Similarity scores:

```json
{
  "results": [
    {
      "document": "ai_in_agriculture.txt",
      "Score": 0.7417751486321952,
      "Snippet": "**AI in Agriculture: Transforming Farming Practices for a Sustainable Future**\n\nArtificial Intelligence (AI) is revolutionizing various sectors, and agriculture is no exception..."
    },
    {
      "document": "ai_in_supply_chain.txt",
      "Score": 0.22212089278254374,
      "Snippet": "**AI in Supply Chain: Transforming Operations and Driving Efficiency**\n\nArtificial Intelligence (AI) has emerged as a transformative force..."
    },
    {
      "document": "ai_and_data_privacy.txt",
      "Score": 0.2070818534642297,
      "Snippet": "**AI and Data Privacy: Navigating the Intersection of Innovation and Protection**\n\nArtificial Intelligence (AI) has emerged as a transformative force..."
    }
  ]
}
```

**Failure Response (400 Bad Request):**

Occurs if you try to search before running the `/index` endpoint.

```json
{
  "Status Code": 400,
  "Response": "Need to upload docs"
}
```

---

## 📄 Notes

* Always run `/index` after adding or modifying documents.
* Ensure that documents are plain text (`.txt`) for proper TF-IDF processing.
* Cosine Similarity
