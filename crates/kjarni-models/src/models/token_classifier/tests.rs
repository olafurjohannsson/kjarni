use super::*;

fn token(label: &str, score: f32, start: usize, end: usize) -> TokenLabel {
    TokenLabel {
        token: String::new(),
        label: label.into(),
        score,
        start,
        end,
    }
}

#[test]
fn b_and_i_tokens_of_one_tag_make_one_entity() {
    let text = "Þórður Jónsson fór";
    let tokens = [
        token("B-Person", 0.9, 0, 6),
        token("I-Person", 0.7, 7, 14),
        token("O", 0.99, 15, 18),
    ];
    let entities = group_entities(text, &tokens);
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].label, "Person");
    assert_eq!(entities[0].text, "Þórður Jónsson");
    assert_eq!((entities[0].start, entities[0].end), (0, 14));
    assert!((entities[0].score - 0.8).abs() < 1e-6);
}

#[test]
fn a_b_tag_starts_a_new_entity_even_after_the_same_tag() {
    let tokens = [token("B-LOC", 1.0, 0, 1), token("B-LOC", 1.0, 2, 3)];
    assert_eq!(group_entities("a b", &tokens).len(), 2);
}

#[test]
fn an_i_tag_after_another_tag_starts_an_entity_of_its_own() {
    let tokens = [token("B-PER", 1.0, 0, 1), token("I-LOC", 0.5, 2, 3)];
    let entities = group_entities("a b", &tokens);
    assert_eq!(
        entities
            .iter()
            .map(|e| e.label.as_str())
            .collect::<Vec<_>>(),
        ["PER", "LOC"]
    );
}

#[test]
fn labels_without_a_prefix_group_while_they_repeat() {
    let tokens = [
        token("PER", 1.0, 0, 3),
        token("PER", 1.0, 4, 7),
        token("O", 1.0, 8, 9),
    ];
    let entities = group_entities("Ann Lee x", &tokens);
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].text, "Ann Lee");
}

#[test]
fn outside_tokens_and_empty_input_give_no_entities() {
    assert!(group_entities("", &[]).is_empty());
    assert!(group_entities("ab", &[token("O", 1.0, 0, 2)]).is_empty());
}
