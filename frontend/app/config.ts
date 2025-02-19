export default process.env.NODE_ENV === "production" ? {
	"backend_uri": "http://backend:5000" // docker network
} : {
	"backend_uri": "http://localhost:5000"
}