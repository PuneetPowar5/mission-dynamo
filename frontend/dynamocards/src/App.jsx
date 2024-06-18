import React, {useState} from 'react'
import axios from 'axios'
import Flashcard from './Flashcard';
import './App.css';

function App () {
    const [youtubeLink, setLink] = useState("");
    const [keyConcepts, setKeyConcepts] = useState([]);

    const handleLinkChange = (event) => {
        setLink(event.target.value);
    }

    const sendLink = async () => {
        try{
            const response = await axios.post("http://localhost:8000/analyze_video", {
                youtube_link: youtubeLink,
            });
            const data = response.data;
            if (data.key_concepts && Array.isArray(data.key_concepts)) {
                const transformedConcepts = data.key_concepts.flatMap(concept => {
                  return Object.entries(concept).map(([term, definition]) => ({
                    term,
                    definition
                  }));
                });
                setKeyConcepts(transformedConcepts);
              }
            else{
                console.error("Data does not contain key concepts tag: ", data);
                setKeyConcepts([]);
            }
        } catch (error){
            console.log(error);
            setKeyConcepts([]);
        }
    };

    const discardFlashcard = (index) => {
        setKeyConcepts(currentConcepts => currentConcepts.filter((_, i) => i !== index));
    }

  return (
    <div className='App'>
        <h1 className='title'>Youtube Link to Flashcards Generator</h1>
        <input className='inputContainer' type='text' placeholder='Paste Youtube Link here' value={youtubeLink} onChange={handleLinkChange} />
        <button onClick={sendLink}>
            Generate Flashcards
        </button>
        <div className='flashcardContainer'>
        {keyConcepts.map((concept, index) => (
    <Flashcard key={index} term={concept.term} definition={concept.definition} onDiscard={() => discardFlashcard(index)} />
))}
        </div>
    </div>
  )
}

export default App;