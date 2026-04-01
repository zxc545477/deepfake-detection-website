const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB Connection
mongoose.connect(
"mongodb+srv://mayuri:mayuriawalwar@cluster0.ba3af8r.mongodb.net/deepfakeDB?retryWrites=true&w=majority",
{
useNewUrlParser: true,
useUnifiedTopology: true
}
)
.then(()=>{
console.log("MongoDB Connected Successfully");
})
.catch((err)=>{
console.log("Database connection error:", err);
});

// Import User Model
const User = require("./models/User");

// Test Route
app.get("/", (req,res)=>{
res.send("Server is working");
});

// Register API
app.post("/register", async (req,res)=>{
try{

const {name,email,password} = req.body;

const newUser = new User({
name,
email,
password
});

await newUser.save();

res.json({message:"User registered successfully"});

}

catch(error){
res.json({message:"Registration error"});
}
});

// Login API
app.post("/login", async (req,res)=>{
try{

const {email,password} = req.body;

const user = await User.findOne({email});

if(!user){
return res.json({message:"User not found"});
}

if(user.password !== password){
return res.json({message:"Wrong password"});
}

res.json({message:"Login successful"});

}

catch(error){
res.json({message:"Login error"});
}
});

// Start Server
app.listen(5000, ()=>{
console.log("Server running on port 5000");
});