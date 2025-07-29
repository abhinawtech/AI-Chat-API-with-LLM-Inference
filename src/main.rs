mod models;
mod routes;

use std::sync::Arc;

use axum::{routing::post, Router};
use routes::chat::chat_handler;
use tokio::signal;
use tracing_subscriber;
use tower_http::{cors::{Any, CorsLayer}, trace::TraceLayer};

use crate::models::chat::{load_chat_model};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // Initialize your ChatModel (you'll need to implement this based on your model setup)
    let chat_model = Arc::new(
        load_chat_model().expect("Failed to load chat model")
    );

     let cors = CorsLayer::new()
        .allow_origin(Any) // Use `Any` for dev; restrict in prod
        .allow_methods([axum::http::Method::POST])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    let app = Router::new()
        .route("/chat", post(chat_handler))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(chat_model);

    let addr = "127.0.0.1:3000";
    println!("🚀 Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
    .await.unwrap();
}

async fn shutdown_signal() {
    // Wait for CTRL+C
    signal::ctrl_c()
        .await
        .expect("failed to install Ctrl+C handler");
    println!("🔻 Shutting down gracefully");
}
