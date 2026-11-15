//! Errors in the shape OpenAI's API returns them, which client libraries parse
//! into their own exceptions: `{"error": {"message", "type", "param", "code"}}`.

use std::fmt::Display;

use axum::Json;
use axum::extract::{FromRequest, Request};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::de::DeserializeOwned;
use serde_json::json;

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub code: Option<&'static str>,
    pub param: Option<&'static str>,
    pub message: String,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            code: None,
            param: None,
            message: message.into(),
        }
    }

    /// A request parameter with a value this server can't use.
    pub fn invalid(param: &'static str, message: impl Into<String>) -> Self {
        Self {
            param: Some(param),
            code: Some("invalid_value"),
            ..Self::bad_request(message)
        }
    }

    /// A request parameter this server doesn't implement. Refused rather than
    /// ignored, so a client never gets an answer to a question it didn't ask.
    pub fn unsupported(param: &'static str, message: impl Into<String>) -> Self {
        Self {
            param: Some(param),
            code: Some("unsupported_parameter"),
            ..Self::bad_request(message)
        }
    }

    pub fn model_not_found(model: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            param: Some("model"),
            code: Some("model_not_found"),
            ..Self::bad_request(format!(
                "the model '{model}' does not exist; GET /v1/models lists the models Kjarni has"
            ))
        }
    }

    pub fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: Some("invalid_api_key"),
            ..Self::bad_request(
                "missing or wrong API key: send it as 'Authorization: Bearer <key>'",
            )
        }
    }

    /// A failure on the server's side, not the request's.
    pub fn internal(error: impl Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "server_error",
            code: None,
            param: None,
            message: error.to_string(),
        }
    }

    pub fn body(&self) -> serde_json::Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "param": self.param,
                "code": self.code,
            }
        })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body())).into_response()
    }
}

/// A JSON body, with a malformed one answered as an [`ApiError`] instead of
/// axum's plain-text rejection.
pub struct ApiJson<T>(pub T);

impl<S, T> FromRequest<S> for ApiJson<T>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(Self(value)),
            Err(rejection) => Err(ApiError::bad_request(rejection.body_text())),
        }
    }
}
