mod models;
mod routes;
mod services; // Add this

use std::sync::Arc;

use axum::{routing::{get, post}, Router};
use routes::chat::{chat_handler, health_handler};
use services::model_service::ModelService;
use tokio::signal;
use tracing_subscriber;
use tower_http::{cors::{Any, CorsLayer}, trace::TraceLayer};

use crate::models::chat::load_chat_model;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // Load model first
    let chat_model = Arc::new(
        load_chat_model().expect("Failed to load chat model")
    );

    // Create model service
    let model_service = Arc::new(
        ModelService::new(chat_model)
            .await
            .expect("Failed to create model service")
    );

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    let app = Router::new()
        .route("/chat", post(chat_handler))
        .route("/health", get(health_handler)) // Add health check
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(model_service);

    let addr = "127.0.0.1:3000";
    println!("🚀 Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await.unwrap();
}

async fn shutdown_signal() {
    signal::ctrl_c()
        .await
        .expect("failed to install Ctrl+C handler");
    println!("🔻 Shutting down gracefully");
}
