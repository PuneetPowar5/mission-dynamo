import React from 'react'
import './App.css';

function Flashcard({term, definition, onDiscard}){
    return(
        <div className='flashcard'>
            <h3 className='term'>
                {term}
            </h3>
            <p className='definition'>{definition}</p>
            <button className='discard' onClick={onDiscard} style={{marginTop: '10px'}}>
                Discard
            </button>
        </div>
    );
}

export default Flashcard;