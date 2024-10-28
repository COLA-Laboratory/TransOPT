import React, { lazy, useEffect } from 'react'
import './App.css';
import { BrowserRouter as Router, Route, Routes, Navigate, Redirect, Switch} from 'react-router-dom'


import { themeChange } from 'theme-change'
import checkAuth from './app/auth';
import initializeApp from './app/init';



// Importing pages
const Layout = lazy(() => import('./containers/Layout'))


// Initializing different libraries
initializeApp()


// Check for login and initialize axios
const token = checkAuth()


function App() {

  useEffect(() => {
    // 👆 daisy UI themes initialization
    themeChange(false)
  }, [])


  return (
    <>
      <Router>
        <Routes>
          <Route path="/app/*" element={<Layout />} />
          <Route path="*" element={<Navigate to="/app/welcome" replace />} />
        </Routes>
      </Router>
    </>
  )
}

export default App
